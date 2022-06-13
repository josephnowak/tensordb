from typing import Dict, Any

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

    def __init__(self, storage: BaseStorage, max_cached_in_dim: int, dim: str):
        self.storage = storage
        self.max_cached_in_dim = max_cached_in_dim
        self.dim = dim
        self._cached_operations = {}
        self._cached_count = 0
        self._clean_cached_operations()

    def _clean_cached_operations(self):
        self._cached_operations = {
            'store': {'new_data': []},
            'append': {'new_data': []},
            'update': {'new_data': []}
        }
        self._cached_count = 0

    def add_operation(self, type_operation: str, new_data: xr.DataArray, parameters: Dict[str, Any]):
        self._cached_count += new_data.sizes[self.dim]
        if type_operation == 'append' and self._cached_operations['store']['new_data']:
            type_operation = 'store'

        self._cached_operations[type_operation].update(parameters)
        self._cached_operations[type_operation]['new_data'].append(new_data)

        if self._cached_count > self.max_cached_in_dim:
            self.execute_operations()

    def execute_operations(self):
        for type_operation in ['store', 'append', 'update']:
            operation = self._cached_operations[type_operation]
            if not operation['new_data']:
                continue
            operation['new_data'] = xr.concat(
                operation['new_data'],
                dim=self.dim
            )
            getattr(self.storage, type_operation)(**operation)

        self._clean_cached_operations()

    def read(self, **kwargs) -> xr.DataArray:
        self.execute_operations()
        return self.storage.read(**kwargs)

    def append(self, new_data: xr.DataArray, **kwargs):
        self.add_operation('append', new_data, kwargs)

    def update(self, new_data: xr.DataArray, **kwargs):
        self.add_operation('update', new_data, kwargs)

    def store(self, new_data: xr.DataArray, **kwargs):
        self._clean_cached_operations()
        self.add_operation('store', new_data, kwargs)

    def close(self):
        self.execute_operations()
