import fsspec
import xarray
import numpy as np
import zarr
import os
import pandas as pd
import json
import time

from typing import Dict, List, Union, Any
from datetime import datetime
from dask.delayed import Delayed
from loguru import logger

from tensordb.file_handlers import BaseStorage
from tensordb.backup_handlers import S3Handler
from tensordb.file_handlers.zarr_handler.utils import (
    get_affected_chunks,
    update_checksums_temp,
    update_checksums,
    merge_local_checksums
)


class ZarrStorage(BaseStorage):
    """
    TODO:
        1) The next versions of zarr will add support for the modification dates of the chunks, that will simplify
            the code of backup, so It is a good idea modify the code after the modification being published
    """

    def __init__(self,
                 name: str = "data",
                 chunks: Dict[str, int] = None,
                 group: str = None,
                 synchronizer: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.chunks = chunks
        self.group = group
        self.synchronizer = None
        if synchronizer is not None:
            if synchronizer == 'process':
                self.synchronizer = zarr.sync.ProcessSynchronizer(self.local_map.root)
            elif synchronizer == 'thread':
                self.synchronizer = zarr.sync.ThreadSynchronizer()
            else:
                raise ValueError(f"{synchronizer} is not a valid option for the synchronizer")

    def store(self,
              new_data: Union[xarray.DataArray, xarray.Dataset],
              encoding: Dict = None,
              compute: bool = True,
              consolidated: bool = False,
              remote: bool = False,
              **kwargs) -> Any:

        path_map = self.backup_path if remote else self.local_map
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
            update_checksums(path_map, get_affected_chunks(path_map, new_data.coords, self.name))
        else:
            update_checksums_temp(path_map, get_affected_chunks(path_map, new_data.coords, self.name))

        return delayed_write

    def append(self,
               new_data: Union[xarray.DataArray, xarray.Dataset],
               compute: bool = True,
               remote: bool = False,
               **kwargs) -> List[Union[None, Delayed]]:

        exist = self.exist(raise_error_missing_backup=False, **kwargs)
        if not exist:
            return self.store(new_data=new_data, remote=remote, compute=compute, **kwargs)

        path_map = self.backup_path if remote else self.local_map

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
                    path_map,
                    append_dim=dim,
                    compute=compute,
                    group=self.group,
                    synchronizer=None if remote else self.synchronizer
                )
            )

        if remote:
            update_checksums(path_map, get_affected_chunks(path_map, new_data.coords, self.name))
        else:
            update_checksums_temp(path_map, get_affected_chunks(path_map, new_data.coords, self.name))

        return delayed_appends

    def update(self,
               new_data: Union[xarray.DataArray, xarray.Dataset],
               remote: bool = False,
               **kwargs):
        """
        TODO: Avoid loading the entire new data in memory
              Using the to_zarr method of xarray and updating in blocks with the region parameter is
              probably a good solution, the only problem is the time that could take to update,
              but I suppose that the block updating is ideal only when the new_data represent a big % of the entire data
        """
        self.exist(raise_error_missing_backup=True, **kwargs)

        if isinstance(new_data, xarray.Dataset):
            new_data = new_data.to_array()

        path_map = self.backup_path if remote else self.local_map

        act_coords = {k: coord.values for k, coord in self.read(remote=remote).coords.items()}

        coords_names = list(act_coords.keys())
        bitmask = np.isin(act_coords[coords_names[0]], new_data.coords[coords_names[0]].values)
        for coord_name in coords_names[1:]:
            bitmask = bitmask & np.isin(act_coords[coord_name], new_data.coords[coord_name].values)[:, None]

        arr = zarr.open(
            fsspec.FSMap(f'{path_map.root}/{self.name}', path_map.fs),
            mode='a',
            synchronizer=None if remote else self.synchronizer
        )
        arr.set_mask_selection(bitmask, new_data.values.ravel())
        if remote:
            update_checksums(path_map, get_affected_chunks(path_map, new_data.coords, self.name))
        else:
            update_checksums_temp(path_map, get_affected_chunks(path_map, new_data.coords, self.name))


    def upsert(self, new_data: Union[xarray.DataArray, xarray.Dataset], **kwargs):
        self.update(new_data, **kwargs)
        self.append(new_data, **kwargs)

    def read(self,
             consolidated: bool = False,
             chunks: Dict = None,
             remote: bool = False,
             **kwargs) -> xarray.DataArray:

        path_map = self.backup_path if remote else self.local_map
        self.exist(raise_error_missing_backup=True, **kwargs)
        return xarray.open_zarr(
            path_map,
            group=self.group,
            consolidated=consolidated,
            chunks=chunks,
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
        TODO:
            1) Add the synchronizer option for this method, this will prevent from uploading a file that
                is being wrote by another process or thread
        """

        if self.backup_map is None:
            return False

        if overwrite_backup:
            files_names = list(self.local_map.keys())
        else:
            files_names = list(json.loads(self.local_map['temp_checksums.json']).keys())

        self.upload_files(files_names)
        merge_local_checksums(self.local_map)

        return True

    def update_from_backup(self,
                           force_update_from_backup: bool = False,
                           **kwargs) -> bool:
        """
        TODO:
            1) Add the synchronizer option for this method, this will prevent from overwriting a file that
                is being used by another process or thread
        """
        if self.backup_map is None:
            return False

        force_update_from_backup = force_update_from_backup | (not self.local_map.fs.exists(self.local_map.root))

        if force_update_from_backup:
            self.download_files(list(self.backup_map.keys()) + [])
        else:
            if self.local_map['last_modification_date.json'] == self.backup_map['last_modification_date.json']:
                return False

            backup_checksums = json.loads(self.backup_map['checksums.json'])
            local_checksums = json.loads(self.local_map['checksums.json'])
            files_to_download = []
            for name in backup_checksums.keys():
                if name not in local_checksums or backup_checksums[name] != local_checksums[name]:
                    files_to_download.append(name)
            self.download_files(files_to_download + ['last_modification_date.json', 'checksums.json'])

        if self.local_map.fs.exists(f'{self.local_map.root}/temp_last_modification_date.json'):
            self.local_map.fs.rm(f'{self.local_map.root}/temp_last_modification_date.json')

        if self.local_map.fs.exists(f'{self.local_map.root}/temp_checksums.json'):
            self.local_map.fs.rm(f'{self.local_map.root}/temp_checksums.json')

        return True

    def exist(self, raise_error_missing_backup: bool = False, **kwargs):
        if self.local_map.fs.exists(f'{self.local_map.root}/.zattrs'):
            return True
        exist = self.update_from_backup(
            force_update_from_backup=True,
            **kwargs
        )
        if raise_error_missing_backup and not exist:
            raise FileNotFoundError(
                f"The file with local path: {self.local_path} "
                f"does not exist and there is not backup in the bucket {self.bucket_name}"
            )
        return exist

    def close(self, **kwargs):
        self.backup(**kwargs)
