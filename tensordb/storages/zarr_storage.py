import os
import fsspec
import xarray
import numpy as np
import orjson
import zarr

from typing import Dict, List, Union, Any, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from tensordb.storages.base_storage import BaseStorage


class ZarrStorage(BaseStorage):
    """
    Storage created for the Zarr files which implement the necessary methods to be used by the TensorClient.

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

    """

    def __init__(self,
                 chunks: Dict[str, int] = None,
                 synchronizer: Union[Literal['process', 'thread'], None] = None,
                 process_synchronizer_path: str = '',
                 **kwargs):
        super().__init__(**kwargs)
        self.synchronizer = None
        if synchronizer == 'process':
            self.synchronizer = zarr.ProcessSynchronizer(process_synchronizer_path)
        elif synchronizer == 'thread':
            self.synchronizer = zarr.ThreadSynchronizer()
        elif synchronizer is not None:
            raise NotImplemented(f"{synchronizer} is not a valid option for the synchronizer")

        self.chunks = chunks

    def store(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            compute: bool = True,
    ) -> xarray.backends.ZarrStore:

        """
        Store the data, the dtype and all the details will depend of what you pass in the new_data
        parameter, internally this method calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        with a 'w' mode using that data.

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be stored

        Returns
        -------

        An xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_

        """
        new_data = self._transform_to_dataset(new_data)
        delayed_write = new_data.to_zarr(
            self.base_map,
            mode='w',
            compute=compute,
            consolidated=True,
            synchronizer=self.synchronizer,
            group=self.group
        )
        return delayed_write

    def append(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
            compute: bool = True,
    ) -> List[xarray.backends.ZarrStore]:

        """
        Append data at the end of a Zarr file (in case that the file does not exist it will call the store method),
        internally it calls the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        for every dimension of your data.

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want to be appended at the end

        Returns
        -------

        A list of xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        method executed in every dimension

        """
        if not self.exist():
            return self.store(new_data=new_data, compute=compute)

        act_coords = {k: coord.values for k, coord in self.read().coords.items()}
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
                    self.base_map,
                    append_dim=dim,
                    compute=compute,
                    synchronizer=self.synchronizer,
                    consolidated=True,
                    group=self.group,
                )
            )

        return delayed_appends

    def update(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
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

        Parameters
        ----------

        new_data: Union[xarray.DataArray, xarray.Dataset]
            This is the data that want

        complete_update_dims: Union[List, str], default = None
            Modify the coords of your new_data based in the coords of the stored array, basically the dims in the
            complete_update_dims are used to reindex new_data and put NaN whenever there are coords of the original
            array that are not in the coords of new_data.

        Returns
        -------

        An xarray.backends.ZarrStore produced by the
        `to_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        """

        act_data = self._transform_to_dataset(self.read(), chunk_data=False)
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

        delayed_write = act_data.to_zarr(
            self.base_map,
            group=self.group,
            compute=compute,
            synchronizer=self.synchronizer,
            region=regions
        )
        return delayed_write

    def upsert(
            self,
            new_data: Union[xarray.DataArray, xarray.Dataset],
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
            self.update(new_data, compute=compute, complete_update_dims=complete_update_dims)
        ]
        delayed_writes.extend(
            self.append(new_data, compute=compute)
        )
        return delayed_writes

    def read(self) -> Union[xarray.DataArray, xarray.Dataset]:
        """
        Read a tensor stored, internally it use
        `open_zarr method <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_.

        Parameters
        ----------

        Returns
        -------

        An xarray.DataArray or xarray.Dataset that allow to read your tensor, that is the same result that you get with
        `open_zarr <http://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html>`_ and then using the '[]'
        with some names or a name
        """

        arr = xarray.open_zarr(
            self.base_map,
            consolidated=True,
            synchronizer=self.synchronizer,
            group=self.group
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

    def exist(self) -> bool:
        """
        Indicate if the tensor exist or not

        Parameters
        ----------

        Returns
        -------
        True if the tensor exist, False if it not exist

        """

        return '.zmetadata' in self.base_map

    def delete_tensor(self):
        """
        Delete the tensor

        Parameters
        ----------

        """
        self.base_map.clear()
