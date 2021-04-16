import xarray
import os

from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Generic


class BaseStorage:
    def __init__(self,
                 path: str,
                 base_path: str = None,
                 **kwargs):
        self.path = path
        self.base_path = base_path
        self.__dict__.update(kwargs)

    @abstractmethod
    def append(self, new_data: Union[xarray.DataArray, xarray.Dataset], **kwargs):
        pass

    @abstractmethod
    def update(self, new_data: Union[xarray.DataArray, xarray.Dataset], **kwargs):
        pass

    @abstractmethod
    def store(self, new_data: Union[xarray.DataArray, xarray.Dataset], **kwargs):
        pass

    @abstractmethod
    def upsert(self, new_data: Union[xarray.DataArray, xarray.Dataset], **kwargs):
        pass

    @abstractmethod
    def read(self, **kwargs) -> xarray.DataArray:
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
    def exist(self, **kwargs):
        pass

    @property
    def local_path(self):
        return os.path.join("" if self.base_path is None else self.base_path, self.path)

