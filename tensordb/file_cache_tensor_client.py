from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Union, Literal, Callable, ContextManager

import dask
import dask.array as da
import more_itertools as mit
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pydantic import validate_arguments
from xarray.backends.common import AbstractWritableDataStore

from tensordb import dag
from tensordb.algorithms import Algorithms
from tensordb.storages import (
    BaseStorage,
    JsonStorage,
    CachedStorage,
    MAPPING_STORAGES
)
from tensordb.storages.mapping import Mapping, NoLock
from tensordb.tensor_definition import TensorDefinition, MethodDescriptor, Definition
from tensordb.utils.method_inspector import get_parameters
from tensordb.utils.tools import (
    groupby_chunks,
    extract_paths_from_formula,
)
from tensordb.tensor_client import TensorClient


class FileCacheTensorClient:
    """
    Parameters
    ----------

    synchronizer_mode: Literal["manual", "automatic"] = "automatic"
        If this option is enable then all the writes on the base_map are also executed on the remote_map

    tensor_lock: ContextManager = None
        If there are multiple instances using the file cache tensor client then it is possible
        that the same file is downloaded from the remote client at the same time causing data corruption.
        If you need a lock at chunk level then use the locks of the Mapping class
    """
    def __init__(
            self,
            remote_client: TensorClient,
            local_client: TensorClient,
            tensor_lock: ContextManager,
            synchronizer_mode: Literal["manual", "automatic"] = "automatic",
    ):
        self.remote_client = remote_client
        self.local_client = local_client
        self.synchronizer_mode = synchronizer_mode
        self.tensor_lock = tensor_lock

    def synchronize_tensor(
            self,
            path: str,
            mode: Literal["fetch", "merge"],
            force: bool = False,
            only_definition: bool = False
    ):
        if not force and self.synchronizer_mode == "manual" and mode == "merge":
            return

        local_client, remote_client = self.local_client, self.remote_client
        if remote_client.exist(path):
            if mode == "merge":
                remote_client, local_client = self.local_client, self.remote_client

            local_definition = local_client.get_tensor_definition(path)
            remote_definition = remote_client.get_tensor_definition(path)
            local_modification_date = pd.Timestamp(local_definition.metadata["modification_date"])
            remote_modification_date = pd.Timestamp(remote_definition.metadata["modification_date"])

            if local_modification_date == remote_modification_date:
                return

            if remote_modification_date > local_modification_date and mode == "merge":
                raise ValueError(f"During the merge process the remote tensor was modified, "
                                 f"this can be produced by an incorrect behaviour of the tensor_lock")

            if local_modification_date > remote_modification_date and self.synchronizer_mode == "manual":
                return

            local_client.upsert_tensor(remote_definition)

        if not only_definition:
            remote_map = remote_client.base_map.sub_map(path)
            local_map = local_client.base_map.sub_map(path)
            remote_map.copy_to_mapping(local_map, remote_modification_date)

    def read(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch")
            return self.local_client.read(path=path, **kwargs)

    def store(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        with self.tensor_lock(path):
            self.update_tensor_metadata(path, {"modified_date": str(pd.Timestamp.now())})
            delayed = self.local_client.store(path=path, **kwargs)
            self.synchronize_tensor(path, "merge")
            return delayed

    def update(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch")
            self.update_tensor_metadata(path, {"modified_date": str(pd.Timestamp.now())})
            delayed = self.local_client.update(path=path, **kwargs)
            self.synchronize_tensor(path, "merge")
            return delayed

    def upsert(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch")
            self.update_tensor_metadata(path, {"modified_date": str(pd.Timestamp.now())})
            delayed = self.local_client.upsert(path=path, **kwargs)
            self.synchronize_tensor(path, "merge")
            return delayed

    def exist(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self.remote_client.exist(path, **kwargs)

    def drop(
            self,
            path: Union[str, TensorDefinition],
            only_local: bool = True,
            **kwargs
    ) -> List[AbstractWritableDataStore]:
        with self.tensor_lock(path):
            if not only_local:
                self.get_client(True).drop(path=path, **kwargs)
            return self.get_client(False).drop(path=path, **kwargs)

    @validate_arguments
    def create_tensor(self, definition: TensorDefinition):
        delayed = self.local_client.create_tensor(definition=definition)
        self.synchronize_tensor(path=path, mode="merge", force=True)
        return delayed

    @validate_arguments
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch", only_definition=True)
            return self.local_client.get_tensor_definition(path)


    @validate_arguments
    def update_tensor_metadata(self, path: str, new_metadata: Dict[str, Any]):
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch", only_definition=True)
            return self.local_client.update_tensor_metadata(path=path)

    def get_all_tensors_definition(self) -> List[TensorDefinition]:
        return self.remote_client.get_all_tensors_definition()

    @validate_arguments
    def delete_tensor(
            self,
            path: str,
            only_data: bool = False,
            only_local: bool = True,
    ) -> Any:
        with self.tensor_lock(path):
            if not only_local:
                self.remote_client.delete_tensor(path, only_data=only_data)
            return self.local_client.delete_tensor(path=path, only_data=only_data)

    @validate_arguments
    def delete_tensors(
            self,
            paths: List[str],
            only_data: bool = False,
            only_local: bool = True
    ):
        with self.tensor_lock(path):
            if not only_local:
                self.remote_client.delete_tensors(paths, only_data=only_data)
            return self.local_client.delete_tensors(paths, only_data=only_data)

    @validate_arguments
    def get_storage(
            self,
            path: Union[str, TensorDefinition]
    ) -> BaseStorage:
        with self.tensor_lock(path):
            self.synchronize_tensor(path, "fetch", only_definition=True)
            return self.local_client.get_storage(path=path)

