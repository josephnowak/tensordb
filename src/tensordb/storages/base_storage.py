from abc import ABC, abstractmethod

import xarray as xr
from obstore.store import ObjectStore


class BaseStorage(ABC):
    """
    Obligatory interface used for the Storage classes created, this defines the abstract methods for every Storage
    and allow their use with TensorClient.

    Parameters
    ----------

    path: str
        Relative path of your tensor, the TensorClient provide this parameter when it creates the Storage

    base_map: Mapping
        It's the same parameter that you send to the :meth:`TensorClient.__init__` (TensorClient send it automatically)

    tmp_map: Mapping
        Temporal location for rewriting the tensor.

    data_names: Union[str, List[str]], default "data"
        Names of the data vars inside your dataset, if the data_names is a str then the system must return an
        xr.DataArray when you read it

    """

    def __init__(
        self,
        ob_store: ObjectStore,
        sub_path: str,
        data_names: str | list[str] = "data",
        **kwargs,
    ):
        self.ob_store = ob_store
        self.sub_path = sub_path
        self.data_names = data_names

    def get_data_names_list(self) -> list[str]:
        return (
            self.data_names if isinstance(self.data_names, list) else [self.data_names]
        )

    def delete_tensor(self):
        """
        Delete the tensor data from the storage

        Parameters
        ----------

        """
        try:
            paths = self.ob_store.list(self.sub_path).collect()
            paths = [path["path"] for path in paths]
            self.ob_store.delete(paths)
        except FileNotFoundError:
            pass

    @abstractmethod
    def append(
        self, new_data: xr.DataArray | xr.Dataset, **kwargs
    ) -> list[xr.backends.common.AbstractWritableDataStore]:
        """
        This abstractmethod must be overwritten to append new_data to an existing file, the way that it append the data
        will depend on the implementation of the Storage. For example :meth:`ZarrStorage.append`
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
        self, new_data: xr.DataArray | xr.Dataset, **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        """
        This abstractmethod must be overwritten to update new_data to an existing file, so it must not insert any new
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
        self, new_data: xr.DataArray | xr.Dataset, **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        """
        This abstractmethod must be overwritten to store new_data to an existing file, so it must create
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
        self, new_data: xr.DataArray | xr.Dataset, **kwargs
    ) -> list[xr.backends.common.AbstractWritableDataStore]:
        """
        This abstractmethod must be overwritten to update and append new_data to an existing file,
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
    def drop(self, coords, **kwargs) -> xr.backends.common.AbstractWritableDataStore:
        """
        Drop coords of the tensor, this can rewrite the hole file depending on the storage

        Parameters
        ----------
            coords: Dict[
        """
        pass

    @abstractmethod
    def read(self, **kwargs) -> xr.DataArray | xr.Dataset:
        """
        This abstractmethod must be overwritten to read an existing file. Reference :meth:`ZarrStorage.read`

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
        This abstractmethod must be overwritten to check if the tensor exist or not.
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
