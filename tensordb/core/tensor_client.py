import xarray
import os
import fsspec

from typing import Dict, List, Any, Union, Tuple
from numpy import nan, array
from pandas import Timestamp
from loguru import logger

from tensordb.file_handlers import (
    ZarrStorage,
    BaseStorage
)
from tensordb.core.utils import internal_actions


class TensorClient:
    """
        TensorClient
        ------------
        It's a kind of SGBD based on files (not necessary the same type of file). It provide a set of basic methods
        that include append, update, store and retrieve data, all these methods are combined with a backup using S3.

        It was designed with the idea of being an inheritable class, so if there is a file that need a special
        treatment, it will be possible to create a new method that handle that specific file

        Notes
        -----
        1) This class does not have any kind of concurrency but of course the internal handler could have

        2) The actual recommend option to handle the files is using the zarr handler class which allow to write and read
        concurrently

        TODO
        ----
        1) Add methods to validate the data, for example should be useful to check the proportion of missing data
            before save the data.

        2) Add more methods to modify the data like bfill or other xarray methods that can be improved when
            appending data.

        3) Separate the logic of the modify data methods to another class or put them in a functions

        4) Enable the max_files_on_disk option, this will allow to establish a maximum number of files that can be
            on disk.

        5) Improve the tensors definition, I need to think in more ideas for this, but at this moment is really good



    """

    def __init__(self,
                 local_base_map: fsspec.FSMap,
                 backup_base_map: fsspec.FSMap = None,
                 max_files_on_disk: int = 0,
                 synchronizer_definitions: str = None,
                 **kwargs):

        self.local_base_map = local_base_map
        self.backup_base_map = backup_base_map
        self.open_base_store: Dict[str, Dict[str, Any]] = {}
        self.max_files_on_disk = max_files_on_disk
        self._tensors_definition = ZarrStorage(
            path='tensors_definition',
            local_base_map=self.local_base_map,
            backup_base_map=self.backup_base_map,
            synchronizer=synchronizer_definitions
        )

    def add_tensor_definition(self, remote: bool = False, **kwargs):
        if not remote:
            self._tensors_definition.update_from_backup()
        self._tensors_definition.set_attrs(remote=remote, **kwargs)
        if not remote:
            self._tensors_definition.backup()

    def get_tensor_definition(self, path, remote: bool = False) -> Dict:
        if not remote:
            self._tensors_definition.update_from_backup()
        tensor_definition_id = os.path.basename(os.path.normpath(path))
        return self._tensors_definition.get_attrs(remote=remote)[tensor_definition_id]

    def _get_handler(self, path: str, tensor_definition: Dict = None) -> BaseStorage:
        handler_settings = self.get_tensor_definition(path) if tensor_definition is None else tensor_definition
        handler_settings = handler_settings.get('handler', {})
        data_handler = handler_settings.get('data_handler', ZarrStorage)(
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

    def _personalize_handler_action(self, path: str, action_type: str, **kwargs):
        tensor_definition = self.get_tensor_definition(path)
        kwargs.update({
            'action_type': action_type,
            'handler': self._get_handler(path=path, tensor_definition=tensor_definition),
            'tensor_definition': tensor_definition
        })

        method_settings = tensor_definition.get(kwargs['action_type'], {})
        if 'personalized_method' in method_settings:
            method = method_settings['personalized_method']
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
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'read'}})

    def append(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'append'}})

    def update(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'update'}})

    def store(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'store'}})

    def backup(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'backup'}})

    def update_from_backup(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'update_from_backup'}})

    def close(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'close'}})

    def delete(self, path: str, **kwargs) -> xarray.DataArray:
        return self._personalize_handler_action(path=path, **{**kwargs, **{'action_type': 'delete'}})

    def exist(self, path: str, **kwargs):
        return self._get_handler(path, **kwargs).exist(**kwargs)

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
