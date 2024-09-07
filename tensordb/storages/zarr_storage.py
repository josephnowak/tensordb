from typing import Dict, List, Union, Any, Literal, Tuple, Hashable

import numpy as np
import xarray as xr
import zarr

from tensordb.storages.base_storage import BaseStorage
from tensordb.storages.lock import PrefixLock
from tensordb.storages.mapping import Mapping


class ZarrStorage(BaseStorage):
    """
    Storage created for the Zarr files which implement the necessary methods to be used by the TensorClient.

    Parameters
    ----------

    chunks: Dict[str, int], default None
        Define the chunks of the Zarr files, read the doc of the Xarray method
        `to_zarr <https://xr.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html>`_
        in the parameter 'chunks' for more details.

    synchronizer: Union[Literal['process', 'thread', 'distributed'], None, PrefixLock], default None
        Depending on the option send it will create a zarr.sync.ThreadSynchronizer or a zarr.sync.ProcessSynchronizer
        for more info read the doc of `Zarr synchronizer <https://zarr.readthedocs.io/en/stable/api/sync.html>`_
        and the Xarray method `to_zarr <https://xr.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_
        in the parameter 'synchronizer'.

    max_unsort_dims_to_rechunk: int, default 1
        If less or equal dimensions than this number needs to be sorted then create a unique
        chunk along the dimension to avoid generating many small chunks that can generate memory issues

    TODO:
        1. Add more examples to the documentation

    """

    def __init__(
        self,
        tmp_map: Mapping,
        chunks: Dict[str, int] = None,
        synchronizer: Union[Literal["process", "thread"], PrefixLock] = None,
        unique_coords: Dict[str, bool] = None,
        sorted_coords: Dict[str, bool] = None,
        encoding: Dict[str, Any] = None,
        synchronize_only_write: bool = False,
        default_unique_coord: bool = True,
        max_unsort_dims_to_rechunk: int = 1,
        **kwargs,
    ):
        super().__init__(tmp_map=tmp_map, **kwargs)

        lock_path = f"{tmp_map.root}/zarr_process_lock"
        if self.base_map.sub_path is not None:
            lock_path = f"{lock_path}/{self.base_map.sub_path}"

        if synchronizer == "process":
            synchronizer = zarr.ProcessSynchronizer(lock_path)
        elif synchronizer == "thread":
            synchronizer = zarr.ThreadSynchronizer()
        elif synchronizer is not None:
            synchronizer = synchronizer(path=lock_path)

        self.chunks = chunks
        if self.chunks is None:
            self.chunks = {}
        self.synchronizer = synchronizer
        self.unique_coords = unique_coords or {}
        self.sorted_coords = sorted_coords or {}
        self.encoding = encoding
        self.synchronize_only_write = synchronize_only_write
        self.default_unique_coord = default_unique_coord
        self.max_unsort_dims_to_rechunk = max_unsort_dims_to_rechunk

    def _keep_unique_coords(self, new_data):
        new_data = new_data.sel(
            {
                k: ~v.duplicated()
                for k, v in new_data.indexes.items()
                if self.unique_coords.get(k, self.default_unique_coord)
            }
        )
        return new_data

    def _keep_sorted_coords(self, new_data):
        if not self.sorted_coords:
            return new_data

        sorted_coords = {
            k: v.sort_values(ascending=self.sorted_coords[k])
            for k, v in new_data.indexes.items()
            if k in self.sorted_coords
        }
        if self.max_unsort_dims_to_rechunk:
            dims_to_change = {
                k: -1
                for k, v in sorted_coords.items()
                if not new_data.indexes[k].equals(v)
            }
            if len(dims_to_change) <= self.max_unsort_dims_to_rechunk:
                new_data = new_data.chunk(**dims_to_change)

        return new_data.sel(sorted_coords)

    def _validate_sorted_append(self, current_coord, append_coord, dim):
        if dim not in self.sorted_coords:
            return True

        if self.sorted_coords[dim]:
            return current_coord[-1] <= append_coord[0]
        return current_coord[-1] >= append_coord[0]

    def _transform_to_dataset(self, new_data, chunk_data: bool = True) -> xr.Dataset:
        if isinstance(new_data, xr.Dataset):
            new_data = new_data[
                self.data_names
                if isinstance(self.data_names, list)
                else [self.data_names]
            ]
        else:
            if isinstance(new_data, xr.DataArray) and isinstance(self.data_names, list):
                raise ValueError(
                    f"The number of data vars is {len(self.data_names)} which indicate "
                    f"that the tensor is a dataset and the new_data received is a xr.DataArray"
                )
            new_data = new_data.to_dataset(name=self.data_names)

        if chunk_data:
            new_data = new_data.chunk(self.chunks)
        return new_data

    def append_preview(
        self, new_data: xr.Dataset, fill_value
    ) -> Tuple[xr.Dataset, Dict[str | Hashable, xr.Dataset], bool]:
        """
        Generates the datasets that must call the to_zarr method with an append_dim.

        Returns
        -------

        It returns a tuple with three elements:

        1. complete_data: It's the expected results after appending the new data.
            This dataset is delayed, and it's used to restore the whole data
             only if the rewrite output is True, and it's also use for testing and debugging purpose

        2. data_to_append: It's a dict whose keys are dimensions and the values are datasets,
            every dataset is the new data that needs to be persisted in Zarr, it is important to
            highlight that if there is more than one key on this dict then the to_zarr method
            must be invoked one time per dataset and in the order of the dimensions

        3. rewrite: It's a bool that indicates if the dataset must be restored or not
            if True then tensordb must store the dataset again to preserve the order,
            and it can only be True if and only if there is an insertion in the middle
            (or at the beginning)

        """
        act_data = self._transform_to_dataset(self.read(), chunk_data=False)
        new_data = self._keep_unique_coords(new_data)
        new_data = self._keep_sorted_coords(new_data)
        new_data = self._transform_to_dataset(new_data, chunk_data=False)

        self.clear_encoding(new_data)

        # Decide if the data needs to be restored due to insertions in the middle
        rewrite = False
        # The data stored here are the new datasets must be rechunked to be aligned
        # with the chunks of the data already stored.
        # Note: The order of insertion is defined by the dims of the data
        data_to_append = {}
        # The slices are used to calculate the correct chunks to align the data_to_append chunks
        slices_to_append = {}
        # The complete_data is a representation of the final array after concatenating the data_to_append
        # to the act_data, and it is also used to calculate the proper chunks for the data_to_append
        complete_data = act_data

        for dim in new_data.dims:
            new_coord = new_data.indexes[dim]
            act_coord = act_data.indexes[dim]
            # new elements on the coord to append
            coord_to_append = new_coord[~new_coord.isin(act_coord)]
            # Validate if there are new coords to append
            # TODO: This condition is blocking the use of duplicated coordinates
            if len(coord_to_append) == 0:
                continue

            # Check if the data that is going to be inserted respects the sort conditions
            # (if exists), in case that no, then a rewrite is necessary
            rewrite |= not self._validate_sorted_append(
                current_coord=act_coord, append_coord=coord_to_append, dim=dim
            )

            slices_to_append[dim] = {
                k: slice(size, None) if k == dim else slice(0, size)
                for k, size in complete_data.sizes.items()
            }

            reindex_coords = {
                k: coord_to_append if k == dim else act_coord
                for k, act_coord in complete_data.coords.items()
            }

            # Reindex the new_data to align it with the complete_data
            data_to_append[dim] = new_data.reindex(
                reindex_coords, fill_value=fill_value
            )

            # Append the data in a delayed way to simulate the appending of the data
            # and generate the proper chunks to match with Zarr
            complete_data = xr.concat(
                [complete_data, data_to_append[dim]], dim=dim, fill_value=fill_value
            )

        # Rechunk the complete_data to make it consistent with the Zarr chunks
        complete_data = xr.Dataset(
            {
                # TODO: On V1 this must be removed in favour of using encoding
                k: v.chunk(act_data[k].encoding["preferred_chunks"])
                for k, v in complete_data.items()
            }
        )

        if rewrite:
            # If rewrite is necessary then resort, drop duplicates on the coords and rechunk it
            complete_data = self._keep_unique_coords(complete_data)
            complete_data = self._keep_sorted_coords(complete_data)
            complete_data = self._transform_to_dataset(complete_data)
            # Not necessary, return an empty dict to avoid confusion
            data_to_append = {}
        else:
            # Rechunk the data_to_append in the dim order based on the chunks of the complete_data
            # after slicing it, this is to avoid calculating the correct chunks by hand.
            data_to_append = {
                dim: data_to_append[dim].chunk(
                    complete_data.isel(**slices_to_append[dim]).chunksizes
                )
                for dim, data in data_to_append.items()
            }

        return complete_data, data_to_append, rewrite

    def update_preview(
        self,
        new_data: xr.Dataset,
        complete_update_dims: str | List[str] | None,
        fill_value: Any,
    ):
        """
        Generates the dataset that must call the to_zarr method with the regions parameter.

        Returns
        -------

        It returns a tuple with two elements:

        1. update_data: The dataset that contains the new and act values that are inside
            the regions boundary defined by new_data. The chunks are already aligned

        2. regions: Region at which the update_data must be inserted in the Zarr store.
        """
        act_data = self._transform_to_dataset(self.read(), chunk_data=False)
        new_data = self._transform_to_dataset(new_data, chunk_data=False)
        new_data = self._keep_unique_coords(new_data)
        new_data = self._keep_sorted_coords(new_data)

        self.clear_encoding(new_data)

        act_coords = {k: coord for k, coord in act_data.coords.items()}

        # The new data must contain only coordinates that are on the act_coords
        new_data = new_data.sel(
            {k: new_data.coords[k].isin(v) for k, v in act_coords.items()}
        )
        if any(size == 0 for size in new_data.sizes.values()):
            return xr.Dataset(), {}

        if complete_update_dims is not None:
            if isinstance(complete_update_dims, str):
                complete_update_dims = [complete_update_dims]

            new_data = new_data.reindex(
                **{
                    dim: coord
                    for dim, coord in act_coords.items()
                    if dim in complete_update_dims
                },
                fill_value=fill_value,
            )

        regions = {}
        for coord_name in act_data.dims:
            act_bitmask = act_coords[coord_name].isin(
                new_data.coords[coord_name].values
            )
            valid_positions = np.nonzero(act_bitmask.values)[0]
            regions[coord_name] = slice(
                np.min(valid_positions), np.max(valid_positions) + 1
            )

        act_data_region = act_data.isel(**regions)

        # The bitmask_arr identifies which cells of the act_data_region needs to be updated
        # with the new_data, the final shape is equal to the act_data_region
        bitmask_arr = None
        for dim, coord in act_data_region.coords.items():
            # Get which coords needs to be updated for this dimension and chunk it properly
            bitmask_coord = coord.isin(new_data.coords[dim]).chunk(
                {dim: act_data_region.chunksizes[dim]}
            )
            if bitmask_arr is None:
                bitmask_arr = bitmask_coord
            else:
                # Concat a 1D array (called A) with another array (call it B) with different dims
                # is going to generate a cartesian product:
                # B.dims[0] x  B.dims[1] x ...  B.dims[N] x A.dims[0]
                bitmask_arr = bitmask_arr & bitmask_coord

        # reindex the new data to be aligned with the region
        new_data = new_data.reindex(act_data_region.coords, fill_value=fill_value)

        # The chunks must match with the chunks of the actual data after applying the region slice
        new_data = new_data.chunk(act_data_region.chunksizes)

        # Only update the corresponding cells
        update_data = new_data.where(bitmask_arr, act_data_region)

        return update_data, regions

    @staticmethod
    def clear_encoding(dataset):
        # TODO: Once https://github.com/pydata/xarray/issues/4380 is fixed delete the temporal solution of encoding
        for arr in dataset.values():
            arr.encoding.clear()
            for dim in arr.dims:
                arr.coords[dim].encoding.clear()

    def store(
        self,
        new_data: Union[xr.DataArray, xr.Dataset],
        compute: bool = True,
        rewrite: bool = False,
        on_tmp: bool = False,
    ) -> Union[xr.backends.ZarrStore, xr.Dataset, xr.DataArray]:
        """
        Store the data, the dtype and all the details will depend on what you pass in the new_data
        parameter, internally this method calls the method
        `to_zarr <https://xarray.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_
        with a 'w' mode using that data.

        Parameters
        ----------

        new_data: Union[xr.DataArray, xr.Dataset]
            This is the data that want to be stored

        compute: bool, default True
            Same meaning that in xarray

        rewrite: bool, default False
            If it is True, it allows to overwrite the tensor using its own data, this can be inefficient due that
            first it has to store the tensor on a temporal location to then write it on the original and delete
            the temporal.
            The compute option is always set as True if the rewrite option is active

        on_tmp: bool, default False
            If this option is enable then the data is only stored on the temporal folder and
            the stored data is returned, this can be useful for operations like dropna which raise
            the computations automatically

        Returns
        -------
        An xr.backends.ZarrStore produced by the method
        `to_zarr <https://xarray.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_

        """
        new_data = self._keep_unique_coords(new_data)
        new_data = self._keep_sorted_coords(new_data)
        new_data = self._transform_to_dataset(new_data)

        self.clear_encoding(new_data)

        if rewrite or on_tmp:
            compute = True
            new_data.to_zarr(
                self.tmp_map,
                mode="w",
                compute=compute,
                consolidated=True,
                encoding=self.encoding,
            )
            new_data = xr.open_zarr(
                self.tmp_map,
                consolidated=True,
            )
            if on_tmp:
                return new_data[self.data_names]

        self.delete_tensor()

        delayed_write = new_data.to_zarr(
            self.base_map,
            mode="w",
            compute=compute,
            consolidated=True,
            group=self.group,
            encoding=self.encoding,
        )

        if rewrite:
            self.tmp_map.rmdir()

        return delayed_write

    def append(
        self,
        new_data: Union[xr.DataArray, xr.Dataset],
        compute: bool = True,
        fill_value: Any = np.nan,
    ) -> List[xr.backends.ZarrStore]:
        """
        Append data at the end of a Zarr file (in case that the file does not exist it will call the store method),
        internally it calls the method
        `to_zarr <https://xr.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_
        for every dimension of your data.

        Parameters
        ----------

        new_data: Union[xr.DataArray, xr.Dataset]
            This is the data that want to be appended at the end

        compute: bool, default True
            Same meaning that in xarray

        fill_value: Any, default np.nan
            The append method can create many empty cells (equivalent to a pandas/xarray concat) so this parameter
            is used to fill determine the data to fill the empty cells created.

        Returns
        -------

        A list of xr.backends.ZarrStore produced by the to_zarr method executed in every dimension

        """

        if not self.exist():
            return [self.store(new_data=new_data, compute=compute)]

        complete_data, data_to_append, rewrite = self.append_preview(
            new_data=new_data, fill_value=fill_value
        )

        if rewrite:
            return [self.store(new_data=complete_data, compute=compute, rewrite=True)]

        # TODO: For some reason there is an error if more than one dim is tried to be append
        #   without a synchronizer even if they are executed one after the other, or apparently
        #   I'm not using properly the bind function of Dask, so for now set to compute as True
        #   if there is two or more dims to append
        if len(data_to_append) > 1 and self.synchronizer is None:
            compute = True

        delayed_appends = []
        for dim in new_data.dims:
            if dim not in data_to_append:
                continue

            delayed_appends.append(
                data_to_append[dim].to_zarr(
                    self.base_map,
                    append_dim=dim,
                    compute=compute,
                    synchronizer=self.synchronizer,
                    consolidated=True,
                    group=self.group,
                    # TODO: Send an MR to Xarray solving the incorrect detection of chunks
                    safe_chunks=False,
                )
            )

        return delayed_appends

    def update(
        self,
        new_data: Union[xr.DataArray, xr.Dataset],
        compute: bool = True,
        complete_update_dims: Union[List[str], str] = None,
        fill_value: Any = np.nan,
    ) -> Union[xr.backends.ZarrStore, None]:
        """
        Replace data on an existing Zarr files based on the new_data, internally calls the method
        `to_zarr <https://xr.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_ using the
        region parameter, so it automatically creates this region based on your new_data, in some
        cases it could even replace all the data in the file even if you only has two coords in your new_data
        this happened due that Xarray only allows to write in contiguous blocks (region)
        (read carefully how the region parameter works in Xarray)

        Parameters
        ----------

        new_data: Union[xr.DataArray, xr.Dataset]
            This is the data that want

        complete_update_dims: Union[List, str], default = None
            Modify the coords of your new_data based in the coords of the stored array, basically the dims in the
            complete_update_dims are used to reindex new_data and put NaN whenever there are coords of the original
            array that are not in the coords of new_data.

        compute: bool, default True
            Same meaning that in xarray

        fill_value: Any, default np.nan

        Returns
        -------

        A xr.backends.ZarrStore produced by the method
        `to_zarr <https://xr.pydata.org/en/stable/generated/xr.Dataset.to_zarr.html>`_
        """

        update_data, regions = self.update_preview(
            new_data=new_data,
            fill_value=fill_value,
            complete_update_dims=complete_update_dims,
        )
        if not regions:
            return None

        delayed_write = update_data.to_zarr(
            self.base_map,
            group=self.group,
            compute=compute,
            synchronizer=self.synchronizer,
            region=regions,
            # This option is safe based on this https://github.com/pydata/xarray/issues/9072
            safe_chunks=False,
        )
        return delayed_write

    def upsert(
        self,
        new_data: Union[xr.DataArray, xr.Dataset],
        compute: bool = True,
        complete_update_dims: Union[List[str], str] = None,
        fill_value: Any = np.nan,
    ) -> List[xr.backends.ZarrStore]:
        """
        Calls the update and then the append method, if the tensor do not exist then it calls the store method

        Returns
        -------
        A list of xr.backends.ZarrStore produced by the append and update methods

        """
        if not self.exist():
            return [self.store(new_data, compute=compute)]

        delayed_writes = [
            self.update(
                new_data, compute=compute, complete_update_dims=complete_update_dims
            )
        ]
        delayed_writes.extend(
            self.append(new_data, compute=compute, fill_value=fill_value)
        )
        delayed_writes = [write for write in delayed_writes if write is not None]
        return delayed_writes

    def drop(self, coords: Dict, compute: bool = True) -> xr.backends.ZarrStore:
        """
        Drop coords of the tensor, this will rewrite the hole tensor using the rewrite option of store

        Parameters
        ----------

        coords: Dict
            Coords that are going to be deleted from the tensor

        compute: bool, default True
            Same meaning that in xarray

        Returns
        -------
        An xr.backends.ZarrStore produced by the store method

        """
        new_data = self.read()
        new_data = new_data.drop_sel(coords)
        return self.store(new_data=new_data, compute=compute, rewrite=True)

    def read(self) -> Union[xr.DataArray, xr.Dataset]:
        """
        Read a tensor stored, internally it uses
        `open_zarr method <https://xr.pydata.org/en/stable/generated/xr.open_zarr.html>`_.

        Parameters
        ----------

        Returns
        -------
        An xr.DataArray or xr.Dataset that allow to read your tensor, that is the same result that you get with
        `open_zarr <https://xr.pydata.org/en/stable/generated/xr.open_zarr.html>`_ and then using the '[]'
        with some names or a name
        """
        try:
            dataset = xr.open_zarr(
                self.base_map,
                consolidated=True,
                synchronizer=None if self.synchronize_only_write else self.synchronizer,
                group=self.group,
            )
            dataset = dataset[self.data_names]
            return dataset
        except KeyError as e:
            raise KeyError(
                f"The data_names {self.data_names} does not exist on the tensor "
                f"located at: {self.base_map.full_path(None)} or the tensor has not been stored yet"
            ) from e

    def exist(self) -> bool:
        """
        Indicate if the tensor exist or not

        Parameters
        ----------

        Returns
        -------
        True if the tensor exist, False if it does not exist

        """
        try:
            self.read()
            return True
        except KeyError:
            return False
