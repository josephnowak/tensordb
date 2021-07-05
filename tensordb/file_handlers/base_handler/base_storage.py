import xarray
import fsspec

from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Generic


class BaseStorage:
    """
    Interface used to guide every of the Storage classes created, this define the abstract methods for every Storage
    and allow their use with TensorClient.

    """
    def __init__(self,
                 path: str,
                 local_base_map: fsspec.FSMap,
                 backup_base_map: fsspec.FSMap,
                 **kwargs):
        self.path = path
        self.local_map: fsspec.FSMap = fsspec.FSMap(f'{local_base_map.root}/{path}', local_base_map.fs)
        self.backup_map: fsspec.FSMap = fsspec.FSMap(f'{backup_base_map.root}/{path}', backup_base_map.fs)

    @abstractmethod
    def append(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            **kwargs
    ) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        append
        """
        pass

    @abstractmethod
    def update(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            **kwargs
    ) -> xarray.backends.common.AbstractWritableDataStore:
        """
        update
        """
        pass

    @abstractmethod
    def store(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            **kwargs
    ) -> xarray.backends.common.AbstractWritableDataStore:
        """
        store
        """
        pass

    @abstractmethod
    def upsert(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            **kwargs
    ) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        upsert
        """
        pass

    @abstractmethod
    def read(self, remote: bool = False, **kwargs) -> xarray.DataArray:
        """
        read
        """
        pass

    @abstractmethod
    def set_attrs(self, remote: bool = False, **kwargs):
        """
        set_attrs
        """
        pass

    @abstractmethod
    def get_attrs(self, remote: bool = False, **kwargs) -> Dict:
        """
        get_attrs
        """
        pass

    @abstractmethod
    def update_from_backup(self, **kwargs):
        """
        update_from_backup
        """
        pass

    @abstractmethod
    def backup(self, **kwargs):
        """
        backup
        """
        pass

    @abstractmethod
    def close(self, **kwargs):
        """
        close
        """
        pass

    @abstractmethod
    def exist(self, on_local: bool, **kwargs):
        """
        exist
        """
        pass

    @abstractmethod
    def delete_file(self, only_local: bool = True, **kwargs):
        """
        delete_file
        """
        pass

    @staticmethod
    def remove_files(path_map, path: str, recursive: bool = False):
        if path in path_map:
            path_map.fs.rm(f'{path_map.root}/{path}', recursive=recursive)



