from typing import Dict, Any, List, Literal

import xarray as xr

from tensordb.storages import BaseStorage


class CachedStorage:
    """
    CachedStorage only contains the basic methods of the Storages: store, read, update, append,
    close and delete_file, but it was designed to allows two things:

        1.  Keep open the FileStorage of a tensor, this is useful for multiple writes on
            the same file because TensorClient has to always read the tensor_definition before open a tensor
            so this take a lot of time if you have to do a lot of writes.

        2.  Cache a fixed number of writes, this allow to reduce the number of small writes to the file.

    You must always call the close method to guaranty that it write all the cached information.

    Internally it calls the corresponding storage method without take care of the tensor_definition, if you want
    to preserve the tensor_definition and you want to do multiples writes use the compute = True option
    that some Storage support (basically use dask.delayed).

    Parameters
    ----------

    storage: BaseStorage
        Storage of the file.

    max_cached_in_dim: int
        Max number of elements of a dim that can be cached

    dim: str
        Dimension used to count the number of element for max_cached_in_dim

    """

    def __init__(
            self,
            storage: BaseStorage,
            max_cached_in_dim: int,
            dim: str,
            sort_dims: List[str] = None,
            merge_cache: bool = False,
            update_logic: Literal["keep_last", "combine_first"] = "combine_first"
    ):
        self.storage = storage
        self.max_cached_in_dim = max_cached_in_dim
        self.dim = dim
        self.sort_dims = sort_dims
        self._cache = {}
        self._cached_count = 0
        self.merge_cache = merge_cache
        self.update_logic = update_logic
        self._clean_cache()

    def _clean_cache(self):
        self._cache = {
            'store': {'new_data': []},
            'append': {'new_data': []},
            'update': {'new_data': []}
        }
        self._cached_count = 0

    def add_operation(self, type_operation: str, new_data: xr.DataArray, parameters: Dict[str, Any]):
        self._cached_count += new_data.sizes[self.dim]
        if type_operation == 'append' and self._cache['store']['new_data']:
            type_operation = 'store'

        self._cache[type_operation].update(parameters)
        if type_operation == "update" and len(self._cache["update"]['new_data']):
            self.merge_updates(new_data)
        else:
            self._cache[type_operation]['new_data'].append(new_data)

        if self._cached_count > self.max_cached_in_dim:
            self.execute_operations()

    def merge_updates(self, new_data):
        data = self._cache["update"]['new_data'][-1]
        if self.update_logic == "keep_last":
            data = data.sel({self.dim: ~data.coords[self.dim].isin(new_data.coords[self.dim])})
        data = new_data.combine_first(data)
        self._cache["update"]['new_data'][-1] = data

    def merge_update_on_append(self):
        append_data = self._cache["append"]["new_data"]
        update_data = self._cache["update"]["new_data"]
        if not isinstance(update_data, list) and not isinstance(append_data, list):
            common_coord = append_data.indexes[self.dim].intersection(update_data.indexes[self.dim])
            if len(common_coord):
                self._cache["append"]["new_data"] = update_data.sel(**{
                    self.dim: common_coord
                }).combine_first(
                    append_data
                )
                update_data = update_data.sel(**{
                    self.dim: ~update_data.coords[self.dim].isin(common_coord)
                })
                self._cache["update"]["new_data"] = []
                if update_data.sizes[self.dim]:
                    self._cache["update"]["new_data"] = update_data

    def execute_operations(self):
        for type_operation in ['store', 'append', 'update']:
            operation = self._cache[type_operation]
            if not operation['new_data']:
                continue
            operation['new_data'] = xr.concat(
                operation['new_data'],
                dim=self.dim
            )
            if self.sort_dims:
                operation['new_data'] = operation['new_data'].sortby(self.sort_dims)

        if self.merge_cache:
            self.merge_update_on_append()

        for type_operation in ['store', 'append', 'update']:
            operation = self._cache[type_operation]
            if isinstance(operation['new_data'], list):
                continue
            try:
                getattr(self.storage, type_operation)(**operation)
            except Exception as e:
                raise Exception(
                    f"Actual index of the data is {operation['new_data'].indexes[self.dim].values} "
                ) from e

        self._clean_cache()

    def read(self, **kwargs) -> xr.DataArray:
        self.execute_operations()
        return self.storage.read(**kwargs)

    def append(self, new_data: xr.DataArray, **kwargs):
        self.add_operation('append', new_data, kwargs)

    def update(self, new_data: xr.DataArray, **kwargs):
        self.add_operation('update', new_data, kwargs)

    def store(self, new_data: xr.DataArray, **kwargs):
        self._clean_cache()
        self.add_operation('store', new_data, kwargs)

    def close(self):
        self.execute_operations()
