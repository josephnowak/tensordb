import os
import fsspec
import xarray
import numpy as np
import orjson
import zarr

from typing import Dict, List, Union, Any, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from tensordb.storages.base_storage import BaseGridBackupStorage
from tensordb.storages.lock import BaseMapLock


class ZarrStorage(BaseGridBackupStorage):
    metadata_extensions = ['.zattrs', '.zgroup', '.zmetadata']
    """
    Storage created for the Zarr files which implement the necessary methods to be used for TensorClient.

    Internally it store a checksum for every chunk created by Zarr, this allow to speed up the backups and check
    the integrity of the data.

    Parameters
    ----------

    chunks: Dict[str, int], default None
        Define the chunks of the Zarr files, read the doc of the Xarray method
        `to_zarr <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'chunks' for more details.

    synchronizer: {'thread', 'process'}, default None
        Depending on the option send it will create a zarr.sync.ThreadSynchronizer or a zarr.sync.ProcessSynchronizer
        for more info read the doc of `Zarr synchronizer <https://zarr.readthedocs.io/en/stable/api/sync.html>`_
        and the Xarray `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'synchronizer'.


    TODO:
        1. Add more examples to the documentation
        2. The next versions of zarr will add support for the modification dates of the chunks, that will simplify
            the code of backup, so It is a good idea modify the code after the modification being published

    """

    def __init__(self,
                 chunks: Dict[str, int] = None,
                 synchronizer: Union[Literal['process', 'thread'], None] = None,
                 process_synchronizer_path: str = '',
                 mapper_synchronizer: BaseMapLock = None,
                 **kwargs):

        if synchronizer == 'process':
            synchronizer = zarr.ProcessSynchronizer(process_synchronizer_path)
        elif synchronizer == 'thread':
            synchronizer = zarr.ThreadSynchronizer()
        elif synchronizer is not None:
            raise NotImplemented(f"{synchronizer} is not a valid option for the synchronizer")

        mapper_synchronizer = None
        # TODO: Add support for process synchronizer
        if isinstance(synchronizer, zarr.ThreadSynchronizer):
            mapper_synchronizer = synchronizer

        super().__init__(mapper_synchronizer=mapper_synchronizer, **kwargs)

        self.synchronizer = synchronizer
        self.chunks = chunks

    def get_dims(self, remote: bool = False) -> List:
        path_map = self.get_path_map(remote)
        data_names = self.get_data_names_list()[0]
        return path_map.get_as_dict('.zmetadata')['metadata'][f'{data_names}/.zattrs']['_ARRAY_DIMENSIONS']

    def get_chunks_size(self, path: str, remote: bool = False) -> Tuple:
        path_map = self.get_path_map(remote)
        return path_map.get_as_dict('.zmetadata')['metadata'][f'{path}/.zarray']['chunks']

    def store(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            compute: bool = True,
            remote: bool = False,
    ) -> xarray.backends.ZarrStore:

        """
        Store the data, the dtype and all the details will depend of what you pass in the new_data
        parameter, internally this method calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        with a 'w' mode using that data.

        This method must be always called first to completly modify the tensor and the backup, once this method
        is used non of the other method will modify the tensor using the backup

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be stored

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        Returns
        -------

        An xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_

        """
        path_map = self.get_path_map(remote)
        new_data = self._transform_to_dataset(new_data)
        delayed_write = new_data.to_zarr(
            path_map.fs,
            group=self.path,
            mode='w',
            compute=compute,
            consolidated=True,
            synchronizer=self.synchronizer
        )
        self.update_backup_metadata(list(path_map.keys()), remote=remote)
        return delayed_write

    def append(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            compute: bool = True,
            remote: bool = False,
    ) -> List[xarray.backends.ZarrStore]:

        """
        Append data at the end of a Zarr file (in case that the file does not exist it will call the store method),
        internally it calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        for every dimension of your data.

        The tensor is not automatically updated using the backup only in four cases:
        1. The remote parameter is equal to True.
        2. The tensor was previously stored using the store method.
        3. The tensor is updated.
        4. There is no backup.

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be appended at the end

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        Returns
        -------

        A list of xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        method executed in every dimension

        """
        if self.exist(False) and self.requiere_local_update(remote):
            self.update_from_backup(force_update=True)

        if not self.exist(on_local=not remote):
            return self.store(new_data=new_data, remote=remote, compute=compute)

        path_map = self.get_path_map(remote)

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
                    path_map.fs,
                    append_dim=dim,
                    compute=compute,
                    group=self.path,
                    synchronizer=self.synchronizer,
                    consolidated=True,
                )
            )

        self.update_backup_metadata(self.get_modified_files(new_data=new_data, remote=remote), remote=remote)
        return delayed_appends

    def update(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            compute: bool = True,
            complete_update_dims: Union[List[str], str] = None,
    ) -> xarray.backends.ZarrStore:
        """
        Replace data on an existing Zarr files based on the new_data, internally calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_ using the
        region parameter, so it automatically create this region based on your new_data, in some
        cases it could even replace all the data in the file even if you only has two coords in your new_data
        this happend due that Xarray only allows to write in contigous blocks (region)
        (read carefully how the region parameter works in Xarray)

        The tensor is not automatically updated using the backup only in three cases:
        1. The remote parameter is equal to True
        2. The tensor was previously stored using the store method.
        3. The tensor is updated.

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want

        remote: bool, default False
            If the value is True indicate that we want to store the data directly in the backup in the other case
            it will store the data in your local path (then you can use the backup method manually)

        complete_update_dims: Union[List, str], default = None
            Modify the coords of your new_data based in the coords of the stored array, basically the dims in the
            complete_update_dims are used to reindex new_data and put NaN whenever there are coords of the original
            array that are not in the coords of new_data.

        Returns
        -------

        An xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        """

        act_data = self._transform_to_dataset(self.read(remote=remote), chunk_data=False)
        new_data = self._transform_to_dataset(new_data)

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

        path_map = self.get_path_map(remote)
        delayed_write = act_data.to_zarr(
            path_map.fs,
            group=self.path,
            compute=compute,
            synchronizer=self.synchronizer,
            region=regions
        )
        self.update_backup_metadata(self.get_modified_files(new_data=new_data, remote=remote), remote=remote)
        return delayed_write

    def upsert(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            remote: bool = False,
            compute: bool = True,
            complete_update_dims: Union[List[str], str] = None,
    ) -> List[xarray.backends.ZarrStore]:
        """
        Calls the update and then the append method

        Returns
        -------
        A list of xarray.backends.ZarrStore produced by the append and update methods

        """
        delayed_writes = [
            self.update(new_data, remote=remote, compute=compute, complete_update_dims=complete_update_dims)
        ]
        delayed_writes.extend(
            self.append(new_data, remote=remote, compute=compute)
        )
        return delayed_writes

    def read(
            self,
            remote: bool = False,
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """
        Read a tensor stored, internally it use
        `open_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_.

        The tensor is not automatically updated using the backup only in three cases:
        1. The remote parameter is equal to True
        2. The tensor was previously stored using the store method.
        3. The tensor is updated.

        Parameters
        ----------

        remote: bool, default False
            If the value is True indicate that we want to read the data directly from the backup in the other case
            it will read the data from your local path if possible, if not it automatically download the backup.
            (This only

        Returns
        -------

        An xarray.DataArray or xarray.Dataset that allow to read your tensor, that is the same result that you get with
        `open_zarr <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_ and then using the '[]'
        with some names or a name
        """

        if self.requiere_local_update(remote):
            self.update_from_backup()

        path_map = self.get_path_map(remote)
        arr = xarray.open_zarr(
            path_map.fs,
            group=self.path,
            consolidated=True,
            synchronizer=self.synchronizer
        )
        return arr[self.data_names]

    def _transform_to_dataset(self, new_data, chunk_data: bool = True) -> xarray.Dataset:
        if isinstance(new_data, xarray.Dataset):
            new_data = new_data[self.data_names]
        else:
            if isinstance(new_data, xarray.DataArray) and isinstance(self.data_names, list):
                raise ValueError(
                    f'The expected number of data vars is {len(self.data_names)} '
                    f'and the new_data has only one (it is an xarray.DataArray)'
                )
            new_data = new_data.to_dataset(name=self.data_names)

        if chunk_data:
            new_data = new_data if self.chunks is None else new_data.chunk(self.chunks)
        return new_data

    def exist(self, on_local: bool) -> bool:
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
        if on_local and ('last_modification_date.json' in self.local_map or
                         'temp_last_modification_date.json' in self.local_map):
            return True

        if not on_local and 'last_modification_date.json' in self.backup_map:
            return True

        return False

    def close(self, **kwargs):
        """
        Close the storage (Zarr does not need this but it will call the backup method automatically)
        """
        self.backup(**kwargs)

    def delete_tensor(self, only_local: bool = True):
        """
        Delete the tensor

        Parameters
        ----------

        only_local: bool, optional True

            Indicate if we want to delete the local path and the backup or only the local,
            True means delete the backup too and False means delete only the local

        """
        self.local_map.clear()
        if not only_local:
            self.backup_map.clear()

    def set_attrs(self, remote: bool = False):
        """
        This is equivalent to use the .attrs of Xarray, so basically is used to add metadata to the tensors,
        these are writed in .zattrs file in json format

        Parameters
        ----------

        remote: bool, default False
            Indicate if we want to write the metadata directly in the backup or in the local path

        """
        path_map = self.get_path_map(remote)
        path_map.update_json('.zattrs', kwargs)
        self.update_backup_metadata(['.zattrs'], remote=remote)

    def get_attrs(self, remote: bool = False):
        """
        Read the metadata of a tensor stored using the set_attrs method

        Parameters
        ----------

        remote: bool, default False
            Indicate if we want to read the metadata directly from the backup or from the local path
        """
        path_map = self.get_path_map(remote)
        return path_map.get_as_dict('.zattrs')

    def get_modified_files(
            self,
            new_data: xarray.DataArray,
            remote: bool = False,
            act_data: xarray.DataArray = None,
    ):
        files_names = super().get_modified_files(new_data=new_data, remote=remote, act_data=act_data)
        files_names.extend(['.zattrs', '.zgroup', '.zmetadata'])
        data_names = self.get_data_names_list()
        files_names.extend([
            path + '/' + extension
            for path in self.get_dims(remote=remote) + data_names
            for extension in ['.zattrs', '.zarray']
        ])
        return files_names
