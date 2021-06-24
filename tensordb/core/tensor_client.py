import loguru
import xarray
import os
import json
import fsspec

from typing import Dict, List, Any, Union, Tuple
from numpy import nan, array
from pandas import Timestamp
from loguru import logger

from tensordb.core.cached_tensor import CachedTensorHandler
from tensordb.file_handlers import (
    ZarrStorage,
    BaseStorage,
    JsonStorage
)
from tensordb.core.utils import internal_actions
from tensordb.config.handlers import mapping_storages


class TensorClient:
    """
        TensorClient
        ------------
        It's a kind of SGBD for tensor data built with Xarray, all the data is stored using Zarr a supported
        Xarray file format (In the future will be more formats) and is organized using a file system storage.

        It provide a set of basic methods: append, update, upsert, store and read, which are customizable and
        can define the complete behaviour of the every tensor, basically you can define triggers for every time you call
        one of those methods.

        Additional features:
            1. Support for any backup system using fsspec package and a specific method to simplify the work (backup).
            2. Creation or modification of new tensors using dynamic string formulas (even string python code)
            3. The read method return a lazy Xarray DataArray instead of only retrieve the data.
            4. It's easy to inherit the class and add customized methods.
            5. The backups are faster and saver because the checksum of every chunk of every tensor
                is stored to avoid uploading or downloading unnecessary data and is useful to check the integrity
                of the data.

        Examples
        --------
        Store and read a dummy array:
            >>> import tensordb
            >>> import fsspec
            >>> import xarray
            >>>
            >>>
            >>> tensor_client = tensordb.TensorClient(
            ...     local_base_map=fsspec.get_mapper('test_db'),
            ...     backup_base_map=fsspec.get_mapper('test_db' + '/backup'),
            ...     synchronizer_definitions='thread'
            ... )
            >>>
            >>> dummy_tensor = xarray.DataArray(
            ...     0,
            ...     coords={'index': list(range(3)), 'columns': list(range(3))},
            ...     dims=['index', 'columns']
            ... )
            >>>
            >>> # Adding a default tensor definition
            >>> tensor_client.add_tensor_definition(dummy_tensor={})
            >>>
            >>> # Storing the dummy tensor
            >>> tensor_client.store(path='dummy_tensor', new_data=dummy_tensor)
            <xarray.backends.zarr.ZarrStore object at 0x00000201E7395A60>
            >>>
            >>> # Reading the dummy tensor (we can avoid the use of path= )
            >>> tensor_client.read(path='dummy_tensor')
            <xarray.DataArray 'data' (index: 3, columns: 3)>
            array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
            Coordinates:
              * columns  (columns) int32 0 1 2
              * index    (index) int32 0 1 2
            >>>

        Storing a tensor from a string formula:
            >>> # Creating a new tensor definition using a formula
            >>> tensor_client.add_tensor_definition(
            ...     dummy_tensor_formula={
            ...        'store': {
            ...             'data_methods': ['read_from_formula'],
            ...         },
            ...         'read_from_formula': {
            ...             'formula': '`dummy_tensor` + 1'
            ...         }
            ...     }
            ... )
            >>>
            >>> # storing the new dummy tensor
            >>> tensor_client.store(path='dummy_tensor_formula')
            <xarray.backends.zarr.ZarrStore object at 0x00000201EA1AB7C0>
            >>>
            >>> # reading the new dummy tensor
            >>> tensor_client.read('dummy_tensor_formula')
            <xarray.DataArray 'data' (index: 3, columns: 3)>
            array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
            Coordinates:
              * columns  (columns) int32 0 1 2
              * index    (index) int32 0 1 2
            >>>

        Appending a new row and a new column to a tensor:
            >>> # Appending a new row and a new columns to a dummy tensor
            >>> new_data = xarray.DataArray(
            ...     2.,
            ...     coords={'index': [3], 'columns': list(range(4))},
            ...     dims=['index', 'columns']
            ... )
            >>>
            >>> tensor_client.append('dummy_tensor_formula', new_data=new_data)
            [<xarray.backends.zarr.ZarrStore object at 0x000001FFF52EBCA0>, <xarray.backends.zarr.ZarrStore object at 0x000001FFF52EBE80>]
            >>> tensor_client.read('dummy_tensor_formula')
            <xarray.DataArray 'data' (index: 4, columns: 4)>
            array([[ 1.,  1.,  1., nan],
                   [ 1.,  1.,  1., nan],
                   [ 1.,  1.,  1., nan],
                   [ 2.,  2.,  2.,  2.]])
            Coordinates:
              * columns  (columns) int32 0 1 2 3
              * index    (index) int32 0 1 2 3
            >>>


        TODO
        ----
        1) Add methods to validate the data, for example should be useful to check the proportion of missing data
            before save the data.

        2) Add more methods to modify the data like bfill or other xarray methods that can be improved when
            appending data.

        3) Separate the logic of the modify data methods to another class or put them in a functions

        4) Enable the max_files_on_disk option, this will allow to establish a maximum number of files that can be
            on disk.

        5) Add documentation and more comments

        Parameters
        ----------
        local_base_map:

    """

    def __init__(self,
                 local_base_map: fsspec.FSMap,
                 backup_base_map: fsspec.FSMap,
                 max_files_on_disk: int = 0,
                 synchronizer: str = None,
                 **kwargs):

        self.local_base_map = local_base_map
        self.backup_base_map = backup_base_map
        self.open_base_store: Dict[str, Dict[str, Any]] = {}
        self.max_files_on_disk = max_files_on_disk
        self.synchronizer = synchronizer
        self._tensors_definition = JsonStorage(
            path='tensors_definition',
            local_base_map=self.local_base_map,
            backup_base_map=self.backup_base_map,
        )

    def add_tensor_definition(self, tensor_id: Union[str, Dict], new_data: Dict):
        self._tensors_definition.store(name=tensor_id, new_data=new_data)

    def create_tensor(self, path: str, tensor_definition: Union[str, Dict], **kwargs):
        json_storage = JsonStorage(path, self.local_base_map, self.backup_base_map)
        kwargs.update({'definition': tensor_definition})
        json_storage.store(new_data=kwargs, name='tensor_definition.json')

    def get_tensor_definition(self, name: str):
        return self._tensors_definition.read(name)

    def get_storage_tensor_definition(self, path) -> Dict:
        json_storage = JsonStorage(path, self.local_base_map, self.backup_base_map)
        if not json_storage.exist('tensor_definition.json'):
            raise KeyError('You can not use a tensor without first call the create_tensor method')
        tensor_definition = json_storage.read('tensor_definition.json')['definition']
        if isinstance(tensor_definition, dict):
            return tensor_definition
        return self.get_tensor_definition(tensor_definition)

    def _get_handler(self, path: str, tensor_definition: Dict = None) -> BaseStorage:
        handler_settings = self.get_storage_tensor_definition(path) if tensor_definition is None else tensor_definition
        handler_settings = handler_settings.get('handler', {})
        handler_settings['synchronizer'] = handler_settings.get('synchronizer', self.synchronizer)

        data_handler = ZarrStorage
        if 'data_handler' in handler_settings:
            data_handler = mapping_storages[handler_settings['data_handler']]

        data_handler = data_handler(
            local_base_map=self.local_base_map,
            backup_base_map=self.backup_base_map,
            path=path,
            **handler_settings
        )
        if path not in self.open_base_store:
            self.open_base_store[path] = {
                'first_read_date': Timestamp.now(),
                'num_use': 0
            }
        self.open_base_store[path]['data_handler'] = data_handler
        self.open_base_store[path]['num_use'] += 1
        return self.open_base_store[path]['data_handler']

    def _customize_handler_action(self, path: str, action_type: str, **kwargs):
        tensor_definition = self.get_storage_tensor_definition(path)
        kwargs.update({
            'action_type': action_type,
            'handler': self._get_handler(path=path, tensor_definition=tensor_definition),
            'tensor_definition': tensor_definition
        })

        method_settings = tensor_definition.get(kwargs['action_type'], {})
        if 'customized_method' in method_settings:
            method = method_settings['customized_method']
            if method in internal_actions:
                return getattr(self, method)(path=path, **kwargs)
            return getattr(self, method)(**kwargs)

        if 'data_methods' in method_settings:
            kwargs['new_data'] = self._apply_data_methods(data_methods=method_settings['data_methods'], **kwargs)

        return getattr(kwargs['handler'], action_type)(**{**kwargs, **method_settings})

    def _apply_data_methods(self,
                            data_methods: List[Union[str, Tuple[str, Dict]]],
                            tensor_definition: Dict,
                            **kwargs):
        results = {**{'new_data': None}, **kwargs}
        for method in data_methods:
            if isinstance(method, (list, tuple)):
                method_name, parameters = method[0], method[1]
            else:
                method_name, parameters = method, tensor_definition.get(method, {})
            result = getattr(self, method_name)(
                **{**parameters, **results},
                tensor_definition=tensor_definition
            )
            if method_name in internal_actions:
                continue

            results.update(result if isinstance(result, dict) else {'new_data': result})

        return results['new_data']

    def read(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'read'}})

    def append(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'append'}})

    def update(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'update'}})

    def store(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'store'}})

    def upsert(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'upsert'}})

    def backup(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'backup'}})

    def update_from_backup(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'update_from_backup'}})

    def set_attrs(self, path: str, **kwargs):
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'set_attrs'}})

    def get_attrs(self, path: str, **kwargs):
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'get_attrs'}})

    def close(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'close'}})

    def delete_file(self, path: str, **kwargs) -> xarray.DataArray:
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'delete_file'}})

    def exist(self, path: str, **kwargs):
        # TODO: this method fail if the tensor was not created, so this must be fixed it should return False
        return self._get_handler(path).exist(**kwargs)

    def exist_tensor_definition(self, path: str):
        base_storage = BaseStorage(path, self.local_base_map, self.backup_base_map)
        return 'tensor_definition.json' in base_storage.backup_map

    def get_cached_tensor_manager(self, path, max_cached_in_dim: int, dim: str, **kwargs):
        handler = self._get_handler(path, **kwargs)
        return CachedTensorHandler(
            file_handler=handler,
            max_cached_in_dim=max_cached_in_dim,
            dim=dim
        )

    def read_from_formula(self,
                          tensor_definition: Dict = None,
                          new_data: xarray.DataArray = None,
                          formula: str = None,
                          use_exec: bool = False,
                          **kwargs):
        if formula is None:
            formula = tensor_definition['read_from_formula']['formula']
            use_exec = tensor_definition['read_from_formula'].get('use_exec', False)

        data_fields = {}
        data_fields_intervals = array([i for i, c in enumerate(formula) if c == '`'])
        for i in range(0, len(data_fields_intervals), 2):
            name_data_field = formula[data_fields_intervals[i] + 1: data_fields_intervals[i + 1]]
            data_fields[name_data_field] = self.read(name_data_field)
        for name, dataset in data_fields.items():
            formula = formula.replace(f"`{name}`", f"data_fields['{name}']")
        if use_exec:
            d = {'data_fields': data_fields, 'new_data': new_data}
            d.update(kwargs)
            exec(formula, d)
            return d['new_data']
        return eval(formula)

    def reindex(self,
                new_data: xarray.DataArray,
                reindex_path: str,
                coords_to_reindex: List[str],
                action_type: str,
                handler: BaseStorage,
                method_fill_value: str = None,
                **kwargs) -> Union[xarray.DataArray, None]:
        if new_data is None:
            return None

        data_reindex = self.read(path=reindex_path, **kwargs)
        if action_type != 'store':
            data = handler.read()
            coords_to_reindex = {
                coord: data_reindex.coords[coord][data_reindex.coords[coord] >= data.coords[coord][-1].values]
                for coord in coords_to_reindex
            }
        else:
            coords_to_reindex = {coord: data_reindex.coords[coord] for coord in coords_to_reindex}
        return new_data.reindex(coords_to_reindex, method=method_fill_value)

    def last_valid_dim(self,
                       new_data: xarray.DataArray,
                       dim: str,
                       **kwargs) -> Union[xarray.DataArray, None]:
        if new_data is None:
            return None
        if new_data.dtype == 'bool':
            return new_data.cumsum(dim=dim).idxmax(dim=dim)
        return new_data.notnull().cumsum(dim=dim).idxmax(dim=dim)

    def replace_values(self,
                       new_data: xarray.DataArray,
                       replace_path: str,
                       value: Any = nan,
                       **kwargs) -> Union[xarray.DataArray, None]:
        if new_data is None:
            return new_data
        replace_data_array = self.read(path=replace_path, **kwargs)
        return new_data.where(replace_data_array.sel(new_data.coords), value)

    def fillna(self,
               new_data: xarray.DataArray,
               value: Any = nan,
               **kwargs) -> Union[xarray.DataArray, None]:

        if new_data is None:
            return new_data
        return new_data.fillna(value)

    def ffill(self,
              handler: BaseStorage,
              new_data: xarray.DataArray,
              dim: str,
              action_type: str,
              limit: int = None,
              **kwargs) -> Union[xarray.DataArray, None]:

        if new_data is None:
            return new_data
        data_concat = new_data
        if action_type != 'store':
            data = handler.read()
            data = data.sel({dim: data.coords[dim] < new_data.coords[dim][0]})
            if data.sizes[dim] > 0:
                data_concat = xarray.concat([data.isel({dim: [-1]}), new_data], dim=dim)

        return data_concat.ffill(dim=dim, limit=limit).sel(new_data.coords)

    def replace_last_valid_dim(self,
                               new_data: xarray.DataArray,
                               replace_path: str,
                               dim: str,
                               value: Any = nan,
                               calculate_last_valid: bool = True,
                               **kwargs) -> Union[xarray.DataArray, None]:
        if new_data is None:
            return new_data

        last_valid = self.read(path=replace_path, **kwargs)
        if calculate_last_valid:
            last_valid = self.last_valid_dim(new_data, dim)
        last_valid = new_data.coords[dim] <= last_valid.fillna(new_data.coords[dim][-1])
        return new_data.where(last_valid.sel(new_data.coords), value)
