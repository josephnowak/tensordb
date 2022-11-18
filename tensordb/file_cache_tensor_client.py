from typing import List, Any, Union, Literal

import pandas as pd
import xarray as xr
from pydantic import validate_arguments
from xarray.backends.common import AbstractWritableDataStore

from tensordb.storages import (
    BaseStorage
)
from tensordb.storages import Mapping, PrefixLock
from tensordb.tensor_client import TensorClient
from tensordb.tensor_definition import TensorDefinition


class FileCacheTensorClient:
    """
    Parameters
    ----------

    synchronizer_mode: Literal["delayed", "automatic"] = "automatic"
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
            tensor_lock: PrefixLock,
            checksum_path: str,
            synchronizer_mode: Literal["delayed", "automatic"] = "automatic",
    ):
        self.remote_client = remote_client
        self.local_client = local_client
        self.synchronizer_mode = synchronizer_mode
        self.tensor_lock = tensor_lock
        self.checksum_map = self.local_client.base_map.sub_map(checksum_path)

    def merge(
            self,
            path: str,
            force: bool = False,
            only_definition: bool = False,
    ):
        if not force and self.synchronizer_mode == "delayed":
            return

        if not self.local_client.exist(path):
            raise ValueError(f"The path {path} does not exist in local so it is impossible to merge it")

        if not self.remote_client.exist(path):
            force = True

        local_definition = self.local_client.get_tensor_definition(path)
        print(path)
        print(local_definition)

        if "modification_date" not in local_definition.metadata:
            raise ValueError(f"There is no modification date on the local definition metadata")

        if not force:
            local_modification_date = pd.Timestamp(local_definition.metadata["modification_date"])
            remote_definition = self.remote_client.get_tensor_definition(path)
            remote_modification_date = pd.Timestamp(remote_definition.metadata["modification_date"])

            if local_modification_date == remote_modification_date:
                return

            if local_modification_date < remote_modification_date:
                raise ValueError(f"The remote tensor was updated last time on {remote_modification_date} "
                                 f"while the local copy was updated on {local_modification_date}, please update "
                                 f"first the local copy before merging it.")

        self.remote_client.upsert_tensor(local_definition)

        if not only_definition:
            Mapping.synchronize(
                remote_map=self.remote_client.base_map.sub_map(path),
                local_map=self.local_client.base_map.sub_map(path),
                checksum_map=self.checksum_map.sub_map(path),
                force=force,
                to_local=False
            )

    def fetch(
            self,
            path: str,
            force: bool = False,
            only_definition: bool = False,
    ):
        if not self.remote_client.exist(path):
            return

        remote_definition = self.remote_client.get_tensor_definition(path)
        if not self.local_client.exist(path):
            force = True

        if "modification_date" not in remote_definition.metadata:
            force = True
            self.remote_client.update_tensor_metadata(path, {"modification_date": str(pd.Timestamp.now())})
            remote_definition = self.remote_client.get_tensor_definition(path)

        if not force:
            remote_modification_date = pd.Timestamp(remote_definition.metadata["modification_date"])
            local_definition = self.local_client.get_tensor_definition(path)
            local_modification_date = pd.Timestamp(local_definition.metadata["modification_date"])

            if local_modification_date == remote_modification_date:
                return

            if local_modification_date > remote_modification_date:
                if self.synchronizer_mode == "delayed":
                    return

                raise ValueError(f"The remote tensor was updated last time on {remote_modification_date} "
                                 f"while the local copy was updated on {local_modification_date}, this means "
                                 f"that they are out of sync and this is not possible "
                                 f"if the synchronizer mode is not delayed")

        self.local_client.upsert_tensor(remote_definition)
        if not only_definition:
            Mapping.synchronize(
                remote_map=self.remote_client.base_map.sub_map(path),
                local_map=self.local_client.base_map.sub_map(path),
                checksum_map=self.checksum_map.sub_map(path),
                force=force,
                to_local=True
            )

    def _exec_callable(
            self,
            func: str,
            path: str,
            fetch: bool,
            merge: bool,
            only_read: bool,
            force: bool = False,
            drop_checksums: bool = False,
            **kwargs

    ):
        func = getattr(self.local_client, func)
        with self.tensor_lock.get_lock(path):
            exist_local = self.local_client.exist(path)
            if fetch:
                self.fetch(path, force=force)
            if not only_read:
                print(self.local_client.get_tensor_definition(path))
                self.local_client.update_tensor_metadata(path, {"modification_date": str(pd.Timestamp.now())})
                print(self.local_client.get_tensor_definition(path))
            if drop_checksums and exist_local:
                print(list(self.checksum_map.sub_map(path).keys()))
                self.checksum_map.rmdir(path)
            result = func(path=path, **kwargs)
            if merge:
                self.merge(path, force=force)
            return result

    def read(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "read",
            path=path,
            fetch=True,
            merge=False,
            only_read=True,
            **kwargs
        )

    def store(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "store",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            drop_checksums=True,
            **kwargs
        )

    def append(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "append",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            **kwargs
        )

    def update(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "update",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            **kwargs
        )

    def upsert(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "upsert",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            **kwargs
        )

    def exist(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> bool:
        return self.remote_client.exist(path, **kwargs)

    def drop(
            self,
            path: Union[str, TensorDefinition],
            **kwargs
    ) -> List[AbstractWritableDataStore]:
        return self._exec_callable(
            "drop",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            **kwargs
        )

    @validate_arguments
    def create_tensor(self, definition: TensorDefinition):
        if self.remote_client.exist(definition.path, only_definition=True):
            raise KeyError(f"Overwrite the tensor definition on the path {definition.path} "
                           f"can produce synchronization problems")
        self.remote_client.create_tensor(definition)
        self.local_client.create_tensor(definition)

    @validate_arguments
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        with self.tensor_lock.get_lock(path):
            self.fetch(path, force=False)
            return self.local_client.get_tensor_definition(path)

    @validate_arguments
    def delete_tensor(
            self,
            path: str,
            only_data: bool = False,
            only_local: bool = False,
    ) -> Any:
        with self.tensor_lock.get_lock(path):
            if self.local_client.exist(path):
                self.local_client.delete_tensor(path=path, only_data=only_data)
            if not only_local:
                self.remote_client.delete_tensor(path=path, only_data=only_data)

    @validate_arguments
    def delete_tensors(
            self,
            paths: List[str],
            only_data: bool = False,
            only_local: bool = True
    ):
        for path in paths:
            try:
                self.delete_tensor(path, only_data=only_data, only_local=only_local)
            except KeyError:
                pass

    @validate_arguments
    def get_storage(
            self,
            path: Union[str, TensorDefinition]
    ) -> BaseStorage:
        return self._exec_callable(
            "get_storage",
            path=path,
            fetch=True,
            merge=False,
            only_read=True,
        )
