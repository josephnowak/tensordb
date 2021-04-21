import xarray
import os
import fsspec

from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Generic
from multiprocessing.pool import ThreadPool


class BaseStorage:
    def __init__(self,
                 path: str,
                 local_base_map: fsspec.FSMap,
                 backup_base_map: fsspec.FSMap = None,
                 **kwargs):
        self.path = path
        self.local_map: fsspec.FSMap = fsspec.FSMap(f'{local_base_map.root}/{path}', local_base_map.fs)
        self.backup_map: fsspec.FSMap = None
        self.max_concurrency = os.cpu_count()
        if backup_base_map is not None:
            self.backup_map = fsspec.FSMap(f'{backup_base_map.root}/{path}', backup_base_map.fs)

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
    def update_from_backup(self, **kwargs):
        pass

    @abstractmethod
    def backup(self, **kwargs):
        pass

    @abstractmethod
    def close(self, **kwargs):
        pass

    @abstractmethod
    def exist(self, remote: bool = False, **kwargs):
        pass

    # def _multi_process_function(self, func: Callable, arguments: List[Dict[str, str]]):
    #     """
    #     TODO: Simplify or improve this code, probably would be better to use map
    #     """
    #     p = ThreadPool(processes=self.max_concurrency)
    #     futures = [p.apply_async(func=func, kwds=kwds) for kwds in arguments]
    #     for future in futures:
    #         future.get()

    def upload_files(self, paths):
        if self.backup_map is None:
            return

        paths = [paths] if isinstance(paths, str) else paths
        for path in paths:
            self.backup_map[path] = self.local_map[path]

    def download_files(self, paths):
        if self.backup_map is None:
            return
        paths = [paths] if isinstance(paths, str) else paths
        for path in paths:
            self.local_map[path] = self.backup_map[path]