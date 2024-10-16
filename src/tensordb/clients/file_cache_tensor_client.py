from typing import List, Any, Union, Literal

import pandas as pd
import xarray as xr
from pydantic import validate_call
from xarray.backends.common import AbstractWritableDataStore

from tensordb.clients.tensor_client import BaseTensorClient, TensorClient
from tensordb.storages import BaseStorage
from tensordb.storages import Mapping, PrefixLock
from tensordb.tensor_definition import TensorDefinition


class FileCacheTensorClient(BaseTensorClient):
    """
    Parameters
    ----------

    synchronizer_mode: Literal["delayed", "automatic"] = "automatic"
        If this option is enable then all the writes on the base_map are also executed on the remote_map

    tensor_lock: PrefixLock = None
        If there are multiple instances using the file cache tensor client then it is possible
        that the same file is downloaded from the remote client at the same time causing data corruption.
        If you need a lock at chunk level then use the locks of the Mapping class

    default_client: Literal["local", "remote"]
        In case that a method is not overwritten by this class then it is going to automatically
        call the method on the default client
    """

    def __init__(
        self,
        remote_client: TensorClient,
        local_client: TensorClient,
        tensor_lock: PrefixLock,
        checksum_path: str,
        synchronizer_mode: Literal["delayed", "automatic"] = "automatic",
        default_client: Literal["local", "remote"] = "remote",
    ):
        self.remote_client = remote_client
        self.local_client = local_client
        self.synchronizer_mode = synchronizer_mode
        self.tensor_lock = tensor_lock
        self.checksum_map = self.local_client.base_map.sub_map(checksum_path)
        self.default_client = remote_client
        if default_client == "local":
            self.default_client = local_client

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self.default_client, item)

    def add_custom_data(self, path, new_data):
        self.remote_client.add_custom_data(path, new_data)

    def get_custom_data(self, path, default=None):
        return self.remote_client.get_custom_data(path, default)

    def upsert_tensor(self, definition: TensorDefinition):
        with self.tensor_lock[definition.path]:
            self.local_client.upsert_tensor(definition)
            self.remote_client.upsert_tensor(definition)

    def get_all_tensors_definition(
        self, include_local: bool = True
    ) -> List[TensorDefinition]:
        definitions = self.remote_client.get_all_tensors_definition()
        if include_local:
            local_definitions = self.local_client.get_all_tensors_definition()
            definitions = {v.path: v for v in definitions}
            local_definitions = {v.path: v for v in local_definitions}
            definitions.update(local_definitions)
            definitions = list(definitions.values())

        return definitions

    @validate_call
    def create_tensor(self, definition: TensorDefinition, keep_metadata: bool = True):
        if self.remote_client.exist(definition.path) and keep_metadata:
            actual_definition = self.remote_client.get_tensor_definition(
                definition.path
            )
            definition.metadata.update(actual_definition.metadata)
        for client in [self.remote_client, self.local_client]:
            client.create_tensor(definition)

    @validate_call
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        return self.remote_client.get_tensor_definition(path)

    @validate_call
    def get_storage(self, path: Union[str, TensorDefinition]) -> BaseStorage:
        path = path.path if isinstance(path, TensorDefinition) else path
        return self._exec_callable(
            "get_storage",
            path=path,
            fetch=True,
            merge=False,
            only_read=True,
            apply_client=self.local_client,
        )

    def merge(
        self,
        path: str,
        force: bool = False,
        only_definition: bool = False,
    ):
        if not force and self.synchronizer_mode == "delayed":
            return

        if not self.local_client.exist(path):
            raise ValueError(
                f"The path {path} does not exist in local so it is impossible to merge it"
            )

        if not self.remote_client.exist(path):
            force = True

        local_definition = self.local_client.get_tensor_definition(path)

        if "modification_date" not in local_definition.metadata:
            raise ValueError(
                f"There is no modification date on the local definition metadata"
            )

        if not force:
            local_modification_date = pd.Timestamp(
                local_definition.metadata["modification_date"]
            )
            remote_definition = self.remote_client.get_tensor_definition(path)
            remote_modification_date = pd.Timestamp(
                remote_definition.metadata["modification_date"]
            )

            if local_modification_date == remote_modification_date:
                return

            if local_modification_date < remote_modification_date:
                raise ValueError(
                    f"The remote tensor was updated last time on {remote_modification_date} "
                    f"while the local copy was updated on {local_modification_date}, please update "
                    f"first the local copy before merging it."
                )

        self.remote_client.create_tensor(local_definition)

        if not only_definition:
            Mapping.synchronize(
                remote_map=self.remote_client.base_map.sub_map(path),
                local_map=self.local_client.base_map.sub_map(path),
                checksum_map=self.checksum_map.sub_map(path),
                force=force,
                to_local=False,
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
            self.remote_client.update_tensor_metadata(
                path, {"modification_date": str(pd.Timestamp.now())}
            )
            remote_definition = self.remote_client.get_tensor_definition(path)

        if not force:
            remote_modification_date = pd.Timestamp(
                remote_definition.metadata["modification_date"]
            )
            local_definition = self.local_client.get_tensor_definition(path)
            local_modification_date = pd.Timestamp(
                local_definition.metadata["modification_date"]
            )

            if local_modification_date == remote_modification_date:
                return

            if local_modification_date > remote_modification_date:
                if self.synchronizer_mode == "delayed":
                    return

                raise ValueError(
                    f"The remote tensor was updated last time on {remote_modification_date} "
                    f"while the local copy was updated on {local_modification_date}, this means "
                    f"that they are out of sync and this is not possible "
                    f"if the synchronizer mode is not delayed"
                )

        self.local_client.create_tensor(remote_definition)
        if not only_definition:
            Mapping.synchronize(
                remote_map=self.remote_client.base_map.sub_map(path),
                local_map=self.local_client.base_map.sub_map(path),
                checksum_map=self.checksum_map.sub_map(path),
                force=force,
                to_local=True,
            )

    def _exec_callable(
        self,
        func: str,
        path: str,
        fetch: bool,
        merge: bool,
        only_read: bool,
        apply_client: TensorClient,
        force: bool = False,
        drop_checksums: bool = False,
        **kwargs,
    ):
        with self.tensor_lock[path]:
            exist_local = self.local_client.exist(path)
            if fetch:
                self.fetch(path, force=force)
            if not only_read:
                self.local_client.update_tensor_metadata(
                    path, {"modification_date": str(pd.Timestamp.now())}
                )
            if drop_checksums and exist_local:
                self.checksum_map.rmdir(path)

            result = getattr(apply_client, func)(path=path, **kwargs)

            if merge:
                self.merge(path, force=force)

        return result

    def read(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "read",
            path=path,
            fetch=True,
            merge=False,
            only_read=True,
            apply_client=self.local_client,
            **kwargs,
        )

    def store(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "store",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            drop_checksums=True,
            apply_client=self.local_client,
            **kwargs,
        )

    def append(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "append",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            apply_client=self.local_client,
            **kwargs,
        )

    def update(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "update",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            apply_client=self.local_client,
            **kwargs,
        )

    def upsert(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        return self._exec_callable(
            "upsert",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            apply_client=self.local_client,
            **kwargs,
        )

    def exist(
        self, path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset], **kwargs
    ) -> bool:
        return self.remote_client.exist(path, **kwargs)

    def drop(
        self, path: Union[str, TensorDefinition], **kwargs
    ) -> List[AbstractWritableDataStore]:
        return self._exec_callable(
            "drop",
            path=path,
            fetch=True,
            merge=True,
            only_read=False,
            apply_client=self.local_client,
            **kwargs,
        )

    @validate_call
    def delete_tensor(
        self,
        path: str,
        only_data: bool = False,
        only_local: bool = False,
    ) -> Any:
        with self.tensor_lock[path]:
            if self.local_client.exist(path):
                self.local_client.delete_tensor(path=path, only_data=only_data)
            if not only_local:
                self.remote_client.delete_tensor(path=path, only_data=only_data)

    @validate_call
    def delete_tensors(
        self, paths: List[str], only_data: bool = False, only_local: bool = True
    ):
        for path in paths:
            try:
                self.delete_tensor(path, only_data=only_data, only_local=only_local)
            except KeyError:
                pass
