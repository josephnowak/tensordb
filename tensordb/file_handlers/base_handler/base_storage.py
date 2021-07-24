import xarray

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Dict, List, Union

from tensordb.utils.sub_mapper import SubMapping


class BaseStorage:
    """
    Obligatory interface used for the Storage classes created, this define the abstract methods for every Storage
    and allow their use with TensorClient.

    Parameters
    ----------

    path: str
        Relative path of your tensor, the TensorClient provide this parameter when it create the Storage

    local_base_map: MutableMapping
        It's the same parameter that you send to the :meth:`TensorClient.__init__` (TensorClient send it automatically)

    backup_base_map: MutableMapping
        It's the same parameter that you send to the :meth:`TensorClient.__init__` (TensorClient send it automatically)

    """
    def __init__(self,
                 path: str,
                 local_base_map: MutableMapping,
                 backup_base_map: MutableMapping,
                 **kwargs):

        self.local_map = SubMapping(
            path=path,
            store=local_base_map
        )
        self.backup_map = SubMapping(
            path=path,
            store=backup_base_map
        )
        self.path = path

    @abstractmethod
    def append(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            **kwargs
    ) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        This abstracmethod must be overwrited to append new_data to an existing file, the way that it append the data
        will depend of the implementation of the Storage. For example :meth:`ZarrStorage.append`
        only append data at the end of the file (probably there will an insert method in the future)

        Parameters
        ----------
        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the tensor that is going to be appended to the stored tensor, it must have the same dims.

        remote: bool, default False
            If the value is True indicate that we want to append the data directly in the backup in the other case
            it will append the data in your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        A list of xarray.backends.common.AbstractWritableDataStore produced by Xarray

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
        This abstracmethod must be overwrited to update new_data to an existing file, so it must not insert any new
        coords, it must only replace elements inside the stored tensor. Reference :meth:`ZarrStorage.update`

        Parameters
        ----------
        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the tensor that is going to be used to update to the stored tensor, it must have the same dims
            and the coords must be subset of the coords of the stored tensor.

        remote: bool, default False
            If the value is True indicate that we want to update the data directly in the backup in the other case
            it will update the data in your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xarray.backends.common.AbstractWritableDataStore produced by Xarray

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
        This abstracmethod must be overwrited to store new_data to an existing file, so it must create
        the necessaries files, folders and metadata for the corresponding tensor. Reference :meth:`ZarrStorage.store`

        Parameters
        ----------
        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the tensor that is going to be stored, the dtypes supported can change depending on the Storage.

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xarray.backends.common.AbstractWritableDataStore produced by Xarray
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
        This abstracmethod must be overwrited to update and append new_data to an existing file,
        so basically it must be a combination between update and append. Reference :meth:`ZarrStorage.upsert`

        Parameters
        ----------
        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the tensor that is going to be upserted in the stored tensor, it must have the same dims.

        remote: bool, default False
            If the value is True indicate that we want to upsert the data directly in the backup in the other case
            it will upsert the data in your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        A list of xarray.backends.common.AbstractWritableDataStore produced by Xarray
        """
        pass

    @abstractmethod
    def read(self, remote: bool = False, **kwargs) -> Union[xarray.DataArray, xarray.Dataset]:
        """
        This abstracmethod must be overwrited to read an existing file. Reference :meth:`ZarrStorage.read`

        Parameters
        ----------

        remote: bool, default False
            If the value is True indicate that we want to read the data directly from the backup in the other case
            it will read the data from your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        An xarray.DataArray or an xarray.Dataset

        """
        pass

    @abstractmethod
    def set_attrs(self, remote: bool = False, **kwargs):
        """
        This abstracmethod must be overwrited to set metadata for a tensor. Reference :meth:`ZarrStorage.set_attrs`

        Parameters
        ----------

        remote: bool, default False
            If the value is True indicate that we want to store the metadata directly in the backup in the other case
            it will store the metadata data in your local path.
            Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        """
        pass

    @abstractmethod
    def get_attrs(self, remote: bool = False, **kwargs) -> Dict:
        """
        This abstracmethod must be overwrited to read the metadata of a tensor. Reference :meth:`ZarrStorage.get_attrs`

        Parameters
        ----------

        remote: bool, default False
            If the value is True indicate that we want to read the metadata directly from the backup in the other case
            it will read the metadata from your local path. Not all the Storage can use this parameter so it is optional

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        A dict containing the requested metadata or all the metadata, depended of the Storage.

        """
        pass

    @abstractmethod
    def update_from_backup(self, **kwargs):
        """
        This abstracmethod must be overwrited to update the tensor using the backup.
        Reference :meth:`ZarrStorage.update_from_backup`

        Parameters
        ----------

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        Depends of the Storage

        """
        pass

    @abstractmethod
    def backup(self, **kwargs):
        """
        This abstracmethod must be overwrited to backup the tensor.
        Reference :meth:`ZarrStorage.backup`

        Parameters
        ----------

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        Depends of the Storage

        """
        pass

    @abstractmethod
    def close(self, **kwargs):
        """
        This abstracmethod must be overwrited to close the tensor.
        Reference :meth:`ZarrStorage.close`

        Parameters
        ----------

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        Depends of the Storage
        """
        pass

    @abstractmethod
    def exist(self, on_local: bool, **kwargs):
        """
        This abstracmethod must be overwrited to check if the tensor exist or not.
        Reference :meth:`ZarrStorage.exist`

        Parameters
        ----------

        on_local: bool
            True if you want to check if the tensor exist on the local path, False if you want to check
            if exist in the backup

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        True if the tensor exist, False if it not exist

        """
        pass

    @abstractmethod
    def delete_file(self, only_local: bool = True, **kwargs):
        """
        This abstracmethod must be overwrited to delete the tensor.
        Reference :meth:`ZarrStorage.delete_file`

        Parameters
        ----------

        only_local: bool, optional True
            Indicate if we want to delete the local path and the backup or only the local,
            True means delete the backup too and False means delete only the local

        **kwargs: Dict
            Optional parameters for the Storage

        Returns
        -------
        Depeneds of the Storage

        """
        pass

    @staticmethod
    def remove_files(path_map, path: str, recursive: bool = False):
        if path in path_map:
            path_map.fs.rm(f'{path_map.root}/{path}', recursive=recursive)



