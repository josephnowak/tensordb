import pandas as pd
import xarray as xr
import itertools
import zarr
import numpy as np
import fsspec

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Dict, List, Union, Tuple, Iterable
from loguru import logger


class BaseStorage:
    """
    Obligatory interface used for the Storage classes created, this define the abstract methods for every Storage
    and allow their use with TensorClient.

    Parameters
    ----------

    path: str
        Relative path of your tensor, the TensorClient provide this parameter when it create the Storage

    base_map: MutableMapping
        It's the same parameter that you send to the :meth:`TensorClient.__init__` (TensorClient send it automatically)

    data_names: Union[str, List[str]], default "data"
        Names of the data vars inside your dataset, if the data_names is a str then the system must return an
        xr.DataArray when you read it

    """

    def __init__(self,
                 base_map: MutableMapping,
                 tmp_map: MutableMapping,
                 data_names: Union[str, List[str]] = "data",
                 path: str = None,
                 **kwargs):
        self.base_map = base_map
        self.data_names = data_names
        self.group = None
        # TODO: Add the option for tmp_map of use a group
        self.tmp_map = tmp_map
        if path is not None:
            self.tmp_map = tmp_map.fs.get_mapper(tmp_map.root + '/' + path)
            root = self._get_root()
            if root == "":
                self.group = path
            else:
                self.base_map = base_map.fs.get_mapper(root + '/' + path)

    def get_base_map_write(self) -> MutableMapping:
        """
        If the base_map has the local_file cache option active it is going to give a KeyError: '.zmetadata' when
        the tensor is being writted, so this method omit the cache of the base_map in case that it exist
        """
        if self.group is not None or not getattr(self.base_map, 'local_file', False):
            return self.base_map

        return self.base_map.fs.fs.get_mapper(self._get_root())

    def get_data_names_list(self) -> List[str]:
        return self.data_names if isinstance(self.data_names, list) else [self.data_names]

    def _get_root(self) -> str:
        try:
            return self.base_map.root
        except AttributeError:
            try:
                return self.base_map.map.root
            except:
                return ""

    @abstractmethod
    def append(
            self,
            new_data: Union[xr.DataArray, xr.Dataset],
            **kwargs
    ) -> List[xr.backends.common.AbstractWritableDataStore]:
        """
        This abstracmethod must be overwrited to append new_data to an existing file, the way that it append the data
        will depend of the implementation of the Storage. For example :meth:`ZarrStorage.append`
        only append data at the end of the file (probably there will an insert method in the future)

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            This is the tensor that is going to be appended to the stored tensor, it must have the same dims.

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        A list of xr.backends.common.AbstractWritableDataStore produced by Xarray

        """
        pass

    @abstractmethod
    def update(
            self,
            new_data: Union[xr.DataArray, xr.Dataset],
            **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        """
        This abstracmethod must be overwrited to update new_data to an existing file, so it must not insert any new
        coords, it must only replace elements inside the stored tensor. Reference :meth:`ZarrStorage.update`

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            This is the tensor that is going to be used to update to the stored tensor, it must have the same dims
            and the coords must be subset of the coords of the stored tensor.

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xr.backends.common.AbstractWritableDataStore produced by Xarray

        """
        pass

    @abstractmethod
    def store(
            self,
            new_data: Union[xr.DataArray, xr.Dataset],
            **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        """
        This abstracmethod must be overwrited to store new_data to an existing file, so it must create
        the necessaries files, folders and metadata for the corresponding tensor. Reference :meth:`ZarrStorage.store`

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            This is the tensor that is going to be stored, the dtypes supported can change depending on the Storage.

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xr.backends.common.AbstractWritableDataStore produced by Xarray
        """
        pass

    @abstractmethod
    def upsert(
            self,
            new_data: Union[xr.DataArray, xr.Dataset],
            **kwargs
    ) -> List[xr.backends.common.AbstractWritableDataStore]:
        """
        This abstracmethod must be overwrited to update and append new_data to an existing file,
        so basically it must be a combination between update and append. Reference :meth:`ZarrStorage.upsert`

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            This is the tensor that is going to be upserted in the stored tensor, it must have the same dims.

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        A list of xr.backends.common.AbstractWritableDataStore produced by Xarray
        """
        pass

    @abstractmethod
    def drop(
            self,
            coords,
            **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        """
        Drop coords of the tensor, this can rewrite the hole file depending on the storage

        Parameters
        ----------
            coords: Dict[
        """
        pass

    @abstractmethod
    def read(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        """
        This abstracmethod must be overwrited to read an existing file. Reference :meth:`ZarrStorage.read`

        Parameters
        ----------
        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xr.DataArray or an xr.Dataset

        """
        pass

    @abstractmethod
    def exist(self, **kwargs) -> bool:
        """
        This abstracmethod must be overwrited to check if the tensor exist or not.
        Reference :meth:`ZarrStorage.exist`

        Parameters
        ----------

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        True if the tensor exist, False if it not exist

        """
        pass

