import xarray

from typing import Dict, List, Any, Union, Tuple
from loguru import logger

from tensordb.file_handlers import BaseStorage


class CachedTensorHandler:
    def __init__(self, file_handler: BaseStorage, max_cached_in_dim: int, dim: str):
        self.file_handler = file_handler
        self.max_cached_in_dim = max_cached_in_dim
        self.dim = dim
        self._cached_operations = []
        self._cached_count = 0

    def add_operation(self, type_operation: str, parameters: Dict[str, Any]):
        self._cached_count += parameters['new_data'].sizes[self.dim]

        if not self._cached_operations or type_operation == 'update':
            parameters['new_data'] = [parameters['new_data']]
            self._cached_operations.append({'type_operation': type_operation, 'parameters': parameters})
            return

        self._cached_operations[-1]['parameters']['new_data'].append(parameters['new_data'])

        if self._cached_count > self.max_cached_in_dim:
            self.execute_operations()

    def execute_operations(self):
        for operation in self._cached_operations:
            operation['parameters']['new_data'] = xarray.concat(
                operation['parameters']['new_data'],
                dim=self.dim
            )
            getattr(self.file_handler, operation['type_operation'])(**operation['parameters'])

        self._cached_count = 0
        self._cached_operations = []

    def read(self, **kwargs) -> xarray.DataArray:
        self.execute_operations()
        return self.file_handler.read(**kwargs)

    def append(self, new_data: xarray.DataArray, **kwargs):
        kwargs['new_data'] = new_data
        self.add_operation('append', kwargs)

    def update(self, new_data: xarray.DataArray, **kwargs):
        kwargs['new_data'] = new_data
        self.add_operation('update', kwargs)

    def store(self, new_data: xarray.DataArray, **kwargs):
        self._cached_count = 0
        self._cached_operations = []
        kwargs['new_data'] = new_data
        self.add_operation('store', kwargs)

    def close(self):
        self.execute_operations()

    def delete_file(self, only_local: bool = True, **kwargs):
        self.file_handler.delete_file(only_local=only_local, **kwargs)
        self._cached_count = 0
        self._cached_operations = []


