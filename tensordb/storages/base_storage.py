import pandas as pd
import xarray
import itertools
import numpy as np

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Dict, List, Union, Tuple

from tensordb.storages.storage_mapper import StorageMapper
from tensordb.storages.lock import BaseMapLock, BaseLock


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
                 dataset_names: Union[str, List[str]] = "data",
                 mapper_synchronizer: BaseMapLock = None,
                 **kwargs):
        self._local_map = StorageMapper(
            path=path,
            fs=local_base_map,
            mapper_synchronizer=mapper_synchronizer
        )
        self._backup_map = StorageMapper(
            path=path,
            fs=backup_base_map,
            mapper_synchronizer=mapper_synchronizer
        )
        self._path = self.local_map.path
        self.dataset_names = dataset_names

    @property
    def local_map(self) -> StorageMapper:
        return self._local_map

    @property
    def backup_map(self) -> StorageMapper:
        return self._backup_map

    @property
    def path(self) -> str:
        return self._path

    def get_path_map(self, remote: bool) -> StorageMapper:
        return self.backup_map if remote else self.local_map

    @abstractmethod
    def get_dims(self, remote: bool = False, **kwargs) -> List:
        pass

    @abstractmethod
    def get_chunks_size(self, path: str, remote: bool = False, **kwargs) -> Tuple:
        pass

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
    def delete_tensor(self, only_local: bool = True, **kwargs):
        """
        This abstracmethod must be overwrited to delete the tensor.
        Reference :meth:`ZarrStorage.delete_tensor`

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


class BaseGridBackupStorage(BaseStorage):
    backup_paths = ['checksums.json', 'last_modification_date.json']

    def is_local_store(self):
        return self.exist(True) and 'last_modification_date.json' not in self.local_map

    def requiere_local_update(self, remote: bool = False):
        return not remote and not self.is_local_store() and not self.is_updated()

    def is_updated(self):
        # if there is no last_modification_date on the backup then it is updated
        if not self.exist(False):
            return True
        # if there is not last_modification_date in local then there is no data so it is not update
        if 'last_modification_date.json' not in self.local_map:
            return False
        last_modification_date = self.backup_map.get_as_dict('last_modification_date.json')['date']
        local_last_modification_date = self.local_map.get_as_dict('last_modification_date.json')['date']
        return local_last_modification_date == last_modification_date

    def update_backup_metadata(
            self,
            modified_files: List[str],
            remote: bool,
    ):
        path_map = self.get_path_map(remote)
        date = str(pd.Timestamp.now())
        checksums = path_map.checksums(modified_files)
        for path, data in zip(self.backup_paths, [checksums, {'date': date}]):
            path_map.update_json(path if remote else f'temp_{path}', data)

    def backup(self, force_backup: bool = False, **kwargs):
        """
        Store the changes that you have done in your local path in the backup, it avoid overwrite unmodified chunks
        based in the checksum stored.

        If you do all your tensor modifications with remote = True, you do not need to call this method.

        Parameters
        ----------

        force_backup: bool, default False
            Indicate if you want to overwrite or not the entiere backup using the data in your local path

        **kwargs: Dict
            Not used for this method

        Returns
        -------

        True if the backup was executed correctly and False if there is no backup_map

        """
        if self.backup_map is None or 'temp_checksums.json' not in self.local_map:
            return False

        force_backup = force_backup or (not self.exist(False)) or self.is_local_store()

        if not force_backup and not self.is_updated():
            raise ValueError(f'The local copy is not updated, please update it first before make a backup')

        files_names = []
        if not force_backup:
            files_names = list(self.local_map.get_as_dict('temp_checksums.json').keys()) + self.backup_paths

        # update the local backup metadata
        for path in self.backup_paths:
            self.local_map.update_json(path, f'temp_{path}')
            del self.local_map[f'temp_{path}']

        if force_backup:
            files_names = list(self.local_map.keys())

        self.upload_files(files_names)
        return True

    def update_from_backup(self, force_update: bool = False, **kwargs):
        """
        Update the data in your local path using the data in the backup, it will only update the chunks
        that has a different checksum.

        Parameters
        ----------

        force_update: bool
            Indicate if we want to overwrite or not all the chunks independently if they have the same checksum
            True for yes, False for no

        **kwargs: Dict
            Not used
        """
        if not self.exist(False):
            return False

        force_update = force_update or self.is_local_store() or (not self.exist(True))

        if not force_update and (self.backup_map is None or self.is_updated()):
            return False

        if force_update:
            files_names = list(self.backup_map.keys())
        else:
            local_checksums = self.local_map.get_as_dict('checksums.json')
            backup_checksums = self.backup_map.get_as_dict('checksums.json')
            files_names = [
                path
                for path, checksum in backup_checksums.items()
                if local_checksums.get(path, '') != checksum
            ]
            files_names.extend(self.backup_paths)

        self.download_files(files_names)

        # del backup local metadata
        self.local_map.delitems([f'temp_{path}' for path in self.backup_paths if f'temp_{path}' in self.local_map])

        return True

    @abstractmethod
    def get_modified_files(
            self,
            new_data: xarray.DataArray,
            remote: bool = False,
            act_data: xarray.DataArray = None,
            metadata_extensions: List[str] = None
    ):
        if act_data is None:
            act_data = self.read(remote=remote)

        path_map = self.get_path_map(remote)
        dims = self.get_dims(remote)
        dataset_names = self.dataset_names if isinstance(self.dataset_names, list) else [self.dataset_names]

        affected_positions = [
            self.find_affected_positions(act_data.coords[dim].values, new_data.coords[dim].values)
            for dim in dims
        ]
        chunks_names = [
            name
            for dataset_name in dataset_names
            for name in self.find_affected_chunks_names(
                affected_positions,
                self.get_chunks_size(dataset_name, remote=remote),
                prefix=dataset_name
            )
        ]
        chunks_names.extend([
            name
            for i, dim in enumerate(dims)
            for name in self.find_affected_chunks_names(
                [affected_positions[i]],
                self.get_chunks_size(dim, remote=remote),
                prefix=dim
            )
        ])

        return chunks_names

    def upload_files(self, paths):
        self.backup_map.transfer_files(from_path_map=self.local_map, paths=paths)

    def download_files(self, paths):
        self.local_map.transfer_files(from_path_map=self.backup_map, paths=paths)

    @staticmethod
    def find_affected_positions(x: np.ndarray, y: np.ndarray):
        sorted_keys = np.argsort(x)
        return sorted_keys[np.searchsorted(x, y, sorter=sorted_keys)]

    @staticmethod
    def find_affected_chunks_names(
            affected_positions: List[np.ndarray],
            chunks: Tuple[int],
            prefix: str = ""
    ) -> List[str]:
        affected_chunks = [np.unique(positions // chunks[i]) for i, positions in enumerate(affected_positions)]
        return [prefix + '/' + ".".join(map(str, chunk)) for chunk in itertools.product(*affected_chunks)]
