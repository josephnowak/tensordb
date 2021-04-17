import xarray
import os
import json

from typing import Dict, List, Any, Union
from numpy import nan
from pandas import Timestamp
from loguru import logger

from tensordb.file_handlers import (
    ZarrStorage,
    BaseStorage
)
from tensordb.backup_handlers import S3Handler


class TensorClient:
    """
        TensorDB
        ----------
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

        3) Add others backups systems, currently the class only work with S3Handler.

        4) Enable the max_files_on_disk option, this will allow to establish a maximum number of files that can be
            on disk.

        5) The tensors definition must be saved on disk and after that a backup is necessary, probably one solution
            for this is create a ZarrStorage for saving all the files settings (this has a lot of limitation due to
            the zarr way to save strings which is basically an array of fixed size) or simple use a json and make a
            manual backup and allow check the last modified date from S3 (create an extra class for this would be ideal)
            or simple use the attrs of Zarr and ZarrStorage.

        6) Add the option to use s3fs for the paths and read directly from the backup.
    """

    def __init__(self,
                 tensors_definition: Dict[str, Dict[str, Any]],
                 base_path: str,
                 backup_handler: S3Handler = None,
                 max_files_on_disk: int = 0,
                 **kwargs):

        self.base_path = os.path.join(base_path, 'tensors_files_storage')
        self._tensors_definition = tensors_definition
        self.open_base_store: Dict[str, Dict[str, Any]] = {}
        self.max_files_on_disk = max_files_on_disk
        self.backup_hander = backup_handler

        self.__dict__.update(**kwargs)

    def add_tensor_definition(self, tensor_definition_id, tensor_definition):
        self._tensors_definition[tensor_definition_id] = tensor_definition

    def get_tensor_definition(self, path) -> Dict:
        tensor_definition_id = os.path.basename(os.path.normpath(path))
        return self._tensors_definition[tensor_definition_id]

    def _get_handler(self, path: Union[str, List], tensor_definition: Dict = None) -> BaseStorage:
        handler_settings = self.get_tensor_definition(path) if tensor_definition is None else tensor_definition
        handler_settings = handler_settings.get('handler', {})
        local_path = self._complete_path(tensor_definition=handler_settings, path=path)
        if local_path not in self.open_base_store:
            self.open_base_store[local_path] = {
                'data_handler': handler_settings.get('data_handler', ZarrStorage)(
                    base_path=self.base_path,
                    path=self._complete_path(tensor_definition=handler_settings, path=path, omit_base_path=True),
                    backup_hander=self.backup_hander,
                    **handler_settings
                ),
                'first_read_date': Timestamp.now(),
                'num_use': 0
            }
        self.open_base_store[local_path]['num_use'] += 1
        return self.open_base_store[local_path]['data_handler']

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
            if method in ['store', 'update', 'append', 'upsert', 'delete', 'backup', 'update_from_backup', 'close']:
                return getattr(self, method_settings['personalized_method'])(path=path, **kwargs)
            return getattr(self, method)(**kwargs)

        if 'data_methods' in method_settings:
            kwargs['new_data'] = self._apply_data_methods(data_methods=method_settings['data_methods'], **kwargs)

        return getattr(kwargs['handler'], action_type)(**{**kwargs, **method_settings})

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

    def _complete_path(self,
                       tensor_definition: Dict,
                       path: Union[List[str], str],
                       omit_base_path: bool = False):

        path = path if isinstance(path, list) else [path]
        if not omit_base_path:
            return os.path.join(self.base_path, tensor_definition.get('extra_path', ''), *path)
        return os.path.join(tensor_definition.get('extra_path', ''), *path)

    def _apply_data_methods(self, data_methods: List[str], tensor_definition: Dict, **kwargs):
        results = {**{'new_data': None}, **kwargs}
        for method in data_methods:
            result = getattr(self, method)(
                **{**tensor_definition.get(method, {}), **results},
                tensor_definition=tensor_definition
            )
            result = result if isinstance(result, dict) else {'new_data': result}
            results.update(result)
        return results['new_data']

    def read_from_formula(self, tensor_definition, new_data: xarray.DataArray = None, **kwargs):
        formula = tensor_definition['read_from_formula']['formula']
        data_fields = {}
        data_fields_intervals = [i for i, c in enumerate(formula) if c == '`']
        for i in range(0, len(data_fields_intervals), 2):
            name_data_field = formula[data_fields_intervals[i] + 1: data_fields_intervals[i + 1]]
            data_fields[name_data_field] = self.read(name_data_field)
        for name, dataset in data_fields.items():
            formula = formula.replace(f"`{name}`", f"data_fields['{name}']")
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
