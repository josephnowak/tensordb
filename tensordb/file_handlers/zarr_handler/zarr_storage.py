import os
import fsspec
import xarray
import numpy as np
import zarr
import json

from typing import Dict, List, Union, Any, Literal
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from tensordb.file_handlers import BaseStorage
from tensordb.file_handlers.zarr_handler.utils import (
    get_affected_chunks,
    update_checksums_temp,
    update_checksums,
    merge_local_checksums,
    get_lock
)


class ZarrStorage(BaseStorage):
    """
    Storage created for the Zarr files which implement the necessary methods to be used for TensorClient.

    Internally it store a checksum for every chunk created by Zarr, this allow to speed up the backups and check
    the integrity of the data.

    Parameters
    ----------

    name: str, default 'data'
        Name of the dataset, this is necessary if you want to work with Xarray DataArrays
        due that Xarray only support writes to Zarr files from datasets.
        Read the doc of `to_dataset <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_dataset.html?highlight=dataset>`_

    chunks: Dict[str, int], default None
        Define the chunks of the Zarr files, read the doc of the Xarray method
        `to_zarr <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'chunks' for more details.

    group: str, default None
        read the doc of the Xarray method
        `to_zarr <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'group' for more details.

    synchronizer: {'thread', 'process'}, default None
        Depending on the option send it will create a zarr.sync.ThreadSynchronizer or a zarr.sync.ProcessSynchronizer
        for more info read the doc of `Zarr synchronizer <https://zarr.readthedocs.io/en/stable/api/sync.html>`_
        and the Xarray `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'synchronizer'.


    TODO:
        1. Add more documentation and better comments
        2. The next versions of zarr will add support for the modification dates of the chunks, that will simplify
            the code of backup, so It is a good idea modify the code after the modification being published

    """

    def __init__(self,
                 name: str = "data",
                 chunks: Dict[str, int] = None,
                 group: str = None,
                 synchronizer: Union[Literal['process', 'thread'], None] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.chunks = chunks
        self.group = group
        self.max_concurrency = os.cpu_count()
        if synchronizer is None:
            self.synchronizer = None
        elif synchronizer == 'process':
            self.synchronizer = zarr.ProcessSynchronizer(self.local_map.root)
        elif synchronizer == 'thread':
            self.synchronizer = zarr.ThreadSynchronizer()
        else:
            raise NotImplemented(f"{synchronizer} is not a valid option for the synchronizer")

    def store(self,
              new_data: Union[xarray.DataArray, xarray.Dataset],
              compute: bool = True,
              consolidated: bool = True,
              remote: bool = False,
              encoding: Dict = None,
              **kwargs) -> xarray.backends.common.AbstractWritableDataStore:

        """
        Store the data in the Zarr file, the dtype and all the details will depend of what you pass in the new_data
        parameter, internally this method calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        with a 'w' mode using that data.

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be stored

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        consolidated: bool, default True
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'consolidated'

        encoding: Dict, default None
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'encoding'

        **kwargs: Dict
            It is not used in this method but due to the way the TensorClient pass the parameters it is necessary

        Returns
        -------

        An xarray.backends.common.AbstractWritableDataStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_

        """
        path_map = self.backup_map if remote else self.local_map
        new_data = self._transform_to_dataset(new_data)
        delayed_write = new_data.to_zarr(
            path_map,
            group=self.group,
            mode='w',
            encoding=encoding,
            compute=compute,
            consolidated=consolidated,
            synchronizer=None if remote else self.synchronizer
        )
        if remote:
            update_checksums(path_map, list(path_map.keys()))
        else:
            update_checksums_temp(path_map, list(path_map.keys()))

        return delayed_write

    def append(self,
               new_data: Union[xarray.DataArray, xarray.Dataset],
               compute: bool = True,
               remote: bool = False,
               consolidated: bool = True,
               encoding: Dict = None,
               **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:

        """
        Append data at the end of a Zarr file (in case that the file does not exist it will call the store method),
        internally it calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        for every dimension of your data

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be appended at the end

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        consolidated: bool, default True
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'consolidated'

        encoding: Dict, default None
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'encoding'

        **kwargs: Dict
            Extra parameters used for the read method that is called internally

        Returns
        -------

        A list of xarray.backends.common.AbstractWritableDataStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        method executed in every dimension

        """

        if not self._exist_download(remote=remote):
            return self.store(new_data=new_data, remote=remote, compute=compute, **kwargs)

        path_map = self.backup_map if remote else self.local_map

        act_coords = {k: coord.values for k, coord in self.read(remote=remote).coords.items()}
        delayed_appends = []

        for dim, new_coord in new_data.coords.items():
            coord_to_append = new_coord[~np.isin(new_coord, act_coords[dim])].values
            if len(coord_to_append) == 0:
                continue

            reindex_coords = {
                k: coord_to_append if k == dim else act_coord
                for k, act_coord in act_coords.items()
            }

            data_to_append = new_data.reindex(reindex_coords)

            act_coords[dim] = np.concatenate([act_coords[dim], coord_to_append])
            delayed_appends.append(
                self._transform_to_dataset(data_to_append).to_zarr(
                    path_map if remote else path_map.root,
                    append_dim=dim,
                    compute=compute,
                    group=self.group,
                    synchronizer=None if remote else self.synchronizer,
                    consolidated=consolidated,
                    encoding=encoding
                )
            )

        affected_chunks = get_affected_chunks(path_map, self.read(remote=remote, **kwargs), new_data.coords, self.name)
        if remote:
            update_checksums(path_map, affected_chunks)
        else:
            update_checksums_temp(path_map, affected_chunks)

        return delayed_appends

    def update(self,
               new_data: Union[xarray.DataArray, xarray.Dataset],
               remote: bool = False,
               compute: bool = True,
               encoding: Dict = None,
               complete_update_dims: Union[List[str], str] = None,
               **kwargs) -> xarray.backends.common.AbstractWritableDataStore:
        """
        Replace data on an existing Zarr files based on the new_data, internally calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_ using the
        region parameter, so it automatically create this region based on your new_data, in some
        cases it could even replace all the data in the file even if you only has two coords in your new_data
        this happend due that Xarray only allows to write in contigous blocks (region)
        (read carefully how the region parameter works in Xarray)

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        consolidated: bool, default True
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'consolidated'

        encoding: Dict, default None
            Read the doc of `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
            in the parameter 'encoding'.

        complete_update_dims: Union[List, str], default = None
            Modify the coords of your new_data based in the coords of the stored array, basically the dims in the
            complete_update_dims are used to reindex new_data and put NaN whenever there are coords of the original
            array that are not in the coords of new_data.

        **kwargs: Dict
            Extra parameters used for the read method that is called internally

        Returns
        -------

        An xarray.backends.common.AbstractWritableDataStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        """
        if not self._exist_download(remote=remote):
            raise OSError(
                f'There is no file in the local path: {self.local_map.root} '
                f'and there is no backup in the remote path: {self.backup_map.root}'
            )

        if isinstance(new_data, xarray.Dataset):
            new_data = new_data.to_array(name=self.name)

        act_data = self.read(remote=remote)
        act_coords = {k: coord for k, coord in act_data.coords.items()}
        if complete_update_dims is not None:
            if isinstance(complete_update_dims, str):
                complete_update_dims = [complete_update_dims]
            new_data = new_data.reindex(
                **{dim: coord for dim, coord in act_coords.items() if dim in complete_update_dims}
            )

        bitmask = True
        regions = {}
        for coord_name in act_data.dims:
            act_bitmask = act_coords[coord_name].isin(new_data.coords[coord_name].values)
            valid_positions = np.nonzero(act_bitmask.values)[0]
            regions[coord_name] = slice(np.min(valid_positions), np.max(valid_positions) + 1)
            bitmask = bitmask & act_bitmask.isel(**{coord_name: regions[coord_name]})

        act_data = act_data.isel(**regions)
        new_data = new_data.reindex(act_data.coords)
        act_data = act_data.where(~bitmask, new_data)

        path_map = self.backup_map if remote else self.local_map
        act_data = self._transform_to_dataset(act_data)
        delayed_write = act_data.to_zarr(
            path_map,
            group=self.group,
            compute=compute,
            encoding=encoding,
            synchronizer=None if remote else self.synchronizer,
            region=regions
        )
        affected_chunks = get_affected_chunks(path_map, self.read(remote=remote, **kwargs), new_data.coords, self.name)
        if remote:
            update_checksums(path_map, affected_chunks)
        else:
            update_checksums_temp(path_map, affected_chunks)

        return delayed_write

    def upsert(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        Calls the update and then the append method

        Returns
        -------
        A list of xarray.backends.common.AbstractWritableDataStore produced by the append and update methods

        """
        delayed_writes = [self.update(new_data, **kwargs)]
        delayed_writes.extend(self.append(new_data, **kwargs))
        return delayed_writes

    def read(self,
             consolidated: bool = True,
             remote: bool = False,
             **kwargs) -> xarray.DataArray:
        """
        Read the tensor stored in the Zarr file, internally it use
        `open_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_.

        Parameters
        ----------

        remote: bool, default False
            If the value is True indicate that we want to read the data directly from the backup in the other case
            it will read the data from your local path

        consolidated: bool, default True
            Read the doc of `open_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_
            in the parameter 'consolidated'

        **kwargs: Dict
            Not used

        Returns
        -------

        An xarray.DataArray that allow to read your tensor, that is the same result that you get with
        `open_zarr <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_ and then using the '[]'
        with a name
        """


        if not self._exist_download(remote=remote):
            raise OSError(
                f'There is no file in the local path: {self.local_map.root} '
                f'and there is no backup in the remote path: {self.backup_map.root}'
            )

        path_map = self.backup_map if remote else self.local_map

        return xarray.open_zarr(
            path_map if remote else path_map.root,
            group=self.group,
            consolidated=consolidated,
            chunks=self.chunks,
            synchronizer=None if remote else self.synchronizer
        )[self.name]

    def _transform_to_dataset(self, new_data) -> xarray.Dataset:

        new_data = new_data
        if isinstance(new_data, xarray.DataArray):
            new_data = new_data.to_dataset(name=self.name)
        new_data = new_data if self.chunks is None else new_data.chunk(self.chunks)
        return new_data

    def backup(self, overwrite_backup: bool = False, **kwargs) -> bool:
        """
        Store the changes that you have done in your local path in the backup, it avoid overwrite unmodified chunks
        based in the checksum stored.

        If you do all your tensor modifications with remote = True, you do not need to call this method.

        Parameters
        ----------

        overwrite_backup: bool, default False
            Indicate if you want to overwrite or not the entiere backup using the data in your local path

        **kwargs: Dict
            Not used for this method

        Returns
        -------

        True if the backup was executed correctly and False if there is no backup_map
        """
        if self.backup_map is None:
            return False
        if overwrite_backup:
            files_names = list(self.local_map.keys()) + [
                'last_modification_date.json',
                'checksums.json'
            ]
        else:
            files_names = []
            if 'temp_checksums.json' in self.local_map:
                files_names.extend(list(json.loads(self.local_map['temp_checksums.json']).keys()))
            files_names.extend([
                'last_modification_date.json',
                'checksums.json'
            ])
        merge_local_checksums(self.local_map)
        self.upload_files(files_names)

        return True

    def update_from_backup(self, force_update_from_backup: bool = False, **kwargs) -> bool:
        """
        Update the data in your local path using the data in the backup, it will only update the chunks
        that has a different checksum.

        Parameters
        ----------

        force_update_from_backup: bool
            Indicate if we want to overwrite or not all the chunks independently if they have the same checksum
            True for yes, False for no

        **kwargs: Dict
            Not used
        """
        if self.backup_map is None:
            return False

        if not self.exist(on_local=False):
            raise ValueError(f'There is no backup in the remote path: {self.backup_map.root}')

        force_update_from_backup = force_update_from_backup | ('last_modification_date.json' not in self.local_map)

        if force_update_from_backup:
            self.download_files(list(self.backup_map.keys()))
        else:
            if self.local_map['last_modification_date.json'] == self.backup_map['last_modification_date.json']:
                return False

            backup_checksums = json.loads(self.backup_map['checksums.json'])
            local_checksums = json.loads(self.local_map['checksums.json'])
            files_to_download = [
                name
                for name, checksum in backup_checksums.items()
                if name not in local_checksums or checksum != local_checksums[name]
            ]
            self.download_files(files_to_download + ['last_modification_date.json', 'checksums.json'])

        self.remove_files(self.local_map, 'temp_last_modification_date.json')
        self.remove_files(self.local_map, 'temp_checksums.json')

        return True

    def exist(self, on_local: bool, **kwargs) -> bool:
        """
        Indicate if the tensor exist or not

        Parameters
        ----------

        on_local: bool
            True if you want to check if the tensor exist on the local path, False if you want to check
            if exist in the backup

        Returns
        -------
        True if the tensor exist, False if it not exist

        """
        if on_local and self.local_map.fs.exists(f'{self.local_map.root}/.zattrs'):
            return True

        if not on_local and self.backup_map.fs.exists(f'{self.backup_map.root}/last_modification_date.json'):
            return True

        return False

    def _exist_download(self, remote: bool) -> bool:
        exist_on_local = self.exist(on_local=True)
        exist_on_remote = self.exist(on_local=False)

        if not exist_on_remote and not exist_on_local:
            return False

        if remote and not exist_on_remote:
            return False

        if not remote and not exist_on_local and exist_on_remote:
            self.update_from_backup(True)

        return True

    def close(self, **kwargs):
        """
        Close the storage (Zarr does not need this but it will call the backup method automatically)
        """
        self.backup(**kwargs)

    def delete_file(self, only_local: bool = True, **kwargs):
        """
        Delete the tensor

        Paramters
        ---------

        only_local: bool, optional True

            Indicate if we want to delete the local path and the backup or only the local,
            True means delete the backup too and False means delete only the local

        """
        if self.local_map.fs.exists(self.local_map.root):
            self.local_map.fs.rm(self.local_map.root, recursive=True)

        if not only_local and self.backup_map.fs.exists(self.backup_map.root):
            self.backup_map.fs.rm(self.backup_map.root, recursive=True)

    def set_attrs(self, remote: bool = False, **kwargs):
        """
        This is equivalent to use the .attrs of Xarray, so basically is used to add metadata to the tensors,
        these are writed in .zattrs file in json format

        Parameters
        ----------

        remote: bool, default False
            Indicate if we want to write the metadata directly in the backup or in the local path

        """
        if not self._exist_download(remote=remote):
            raise OSError(
                f'There is no file in the local path: {self.local_map.root} '
                f'and there is no backup in the remote path: {self.backup_map.root}'
            )
        path_map = self.backup_map if remote else self.local_map
        with get_lock(self.synchronizer, '.zattrs'):
            total_attrs = {}
            if '.zattrs' in path_map:
                total_attrs = json.loads(path_map['.zattrs'])
            total_attrs.update(kwargs)
            path_map['.zattrs'] = json.dumps(total_attrs).encode('utf-8')
        if remote:
            update_checksums(path_map, ['.zattrs'])
        else:
            update_checksums_temp(path_map, ['.zattrs'])

    def get_attrs(self, remote: bool = False):
        """
        Read the metadata of a tensor stored using the set_attrs method

        Parameters
        ----------

        remote: bool, default False
            Indicate if we want to read the metadata directly from the backup or from the local path
        """
        if not self._exist_download(remote=remote):
            raise OSError(
                f'There is no file in the local path: {self.local_map.root} '
                f'and there is no backup in the remote path: {self.backup_map.root}'
            )
        path_map = self.backup_map if remote else self.local_map
        return json.loads(path_map['.zattrs'])

    def transfer_files(self, receiver_path_map, sender_path_map, paths):
        paths = [paths] if isinstance(paths, str) else paths
        if len(paths) == 0:
            return

        def transfer(path, lock):
            with lock:
                receiver_path_map[path] = sender_path_map[path]

        with ThreadPoolExecutor(min(os.cpu_count(), len(paths))) as executor:
            executor.map(lambda x: transfer(x, get_lock(self.synchronizer, x)), tuple(paths))

    def upload_files(self, paths):
        if self.backup_map is None:
            return
        self.transfer_files(self.backup_map, self.local_map, paths)

    def download_files(self, paths):
        if self.backup_map is None:
            return
        self.transfer_files(self.local_map, self.backup_map, paths)


