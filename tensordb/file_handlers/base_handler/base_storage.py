import xarray
import fsspec

from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Generic


class BaseStorage:
    def __init__(self,
                 path: str,
                 local_base_map: fsspec.FSMap,
                 backup_base_map: fsspec.FSMap,
                 **kwargs):
        self.path = path
        self.local_map: fsspec.FSMap = fsspec.FSMap(f'{local_base_map.root}/{path}', local_base_map.fs)
        self.backup_map: fsspec.FSMap = fsspec.FSMap(f'{backup_base_map.root}/{path}', backup_base_map.fs)

    @abstractmethod
    def append(self, new_data: Union[xarray.DataArray, xarray.Dataset], remote: bool = False, **kwargs):
        pass

    @abstractmethod
    def update(self, new_data: Union[xarray.DataArray, xarray.Dataset], remote: bool = False, **kwargs):
        pass

    @abstractmethod
    def store(self, new_data: Union[xarray.DataArray, xarray.Dataset], remote: bool = False, **kwargs):
        pass

    @abstractmethod
    def upsert(self, new_data: Union[xarray.DataArray, xarray.Dataset], remote: bool = False, **kwargs):
        pass

    @abstractmethod
    def read(self, remote: bool = False, **kwargs) -> xarray.DataArray:
        pass

    @abstractmethod
    def set_attrs(self, remote: bool = False, **kwargs) -> xarray.DataArray:
        pass

    @abstractmethod
    def get_attrs(self, remote: bool = False, **kwargs) -> xarray.DataArray:
        pass

    @abstractmethod
    def update_from_backup(self, **kwargs):
        pass

    @abstractmethod
    def backup(self, **kwargs):
        pass

    @abstractmethod
    def close(self, **kwargs):
        pass

    @abstractmethod
    def exist(self, on_local: bool, **kwargs):
        pass

    @abstractmethod
    def delete_file(self, only_local: bool = True, **kwargs):
        pass

    @staticmethod
    def remove_files(path_map, path: str, recursive: bool = False):
        if path in path_map:
            path_map.fs.rm(f'{path_map.root}/{path}', recursive=recursive)



