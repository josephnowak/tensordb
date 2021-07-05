import loguru
import xarray
import os
import json
import fsspec

from typing import Dict, List, Any, Union, Tuple
from numpy import nan, array
from pandas import Timestamp
from loguru import logger
from dask.delayed import Delayed

from tensordb.core.cached_tensor import CachedTensorHandler
from tensordb.file_handlers import (
    ZarrStorage,
    BaseStorage,
    JsonStorage
)
from tensordb.core.utils import internal_actions
from tensordb.config.handlers import MAPPING_STORAGES


class TensorClient:
    """
    TensorClient
    ------------

    It's client designed to handle tensor data in a simpler way and it's built with Xarray,
    it can support the same files than Xarray but those formats need to be implement
    using the `BaseStorage` interface proposed in this package.

    As we can create Tensors with multiple Storage that needs differents settings or parameters we must create
    a "Tensor Definition" which is basically a dictionary that specify the behaviour that you want to have
    every time you call a method of one Storage. The definitions are very simple to create (there are few internal keys)
    , you only need to use as key the name of your method and as value a dictionary containing all the necessary
    parameters to use it, you can see some examples in the ``Examples`` section

    Additional features:
        1. Support for any backup system using fsspec package and a specific method to simplify the work (backup).
        2. Creation or modification of new tensors using dynamic string formulas (even string python code).
        3. The read method return a lazy Xarray DataArray instead of only retrieve the data.
        4. It's easy to inherit the class and add customized methods.
        5. The backups can be faster and saver because you can modify them as you want, an example of this is the
           ZarrStorage which has a checksum of every chunk of every tensor stored to
           avoid uploading or downloading unnecessary data and is useful to check the integrity of the data.

    Parameters
    ----------
    local_base_map: fsspec.FSMap
       FSMap instaciated with the local path that you want to use to store all tensors.

    backup_base_map: fsspec.FSMap
        FSMap instaciated with the backup path that you want to use to store all tensors.

    synchronizer: str
        Some of the Storages used to handle the files support a synchronizer, this parameter is used as a default
        synchronizer option for everyone of them (you can pass different synchronizer to every tensor).

    **kwargs: Dict
        Useful when you want to inherent from this class.

    Examples
    --------
    Update this examples.

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

    def add_tensor_definition(self, tensor_id: str, new_data: Dict) -> Dict:
        """
        Add (store) a new tensor definition (internally is stored as a JSON file).

        Parameters
        ----------
        tensor_id: str
            name used to identify the tensor definition.

        new_data: Dict
            Description of the definition, find an example of the format here.

        """
        self._tensors_definition.store(name=tensor_id, new_data=new_data)

    def create_tensor(self, path: str, tensor_definition: Union[str, Dict], **kwargs):
        """
        Create the path and the first file of the tensor which store the necessary metadata to use it,
        this method must always be called before start to write in the tensor.

        Parameters
        ----------
        path: str
            Indicate the location where your tensor is going to be allocated.

        tensor_definition: str, Dict
            This can be an string which allow to read a previously created tensor definition or a completly new
            tensor_definition in case that you pass a Dict.

        **kwargs: Dict
            Aditional metadata for the tensor.

        """
        json_storage = JsonStorage(path, self.local_base_map, self.backup_base_map)
        kwargs.update({'definition': tensor_definition})
        json_storage.store(new_data=kwargs, name='tensor_definition.json')

    def get_tensor_definition(self, name: str) -> Dict:
        """
        Retrieve a created tensor definition.

        Parameters
        ----------
        name: str
            name of the tensor definition.

        Returns
        -------
        A dict containing all the information of the tensor definition previusly stored.

        """
        return self._tensors_definition.read(name)

    def get_storage_tensor_definition(self, path) -> Dict:
        """
        Retrieve the tensor definition of an specific tensor.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        Returns
        -------
        A dict containing all the information of the tensor definition previusly stored.

        """
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
            data_handler = MAPPING_STORAGES[handler_settings['data_handler']]

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

    def storage_method_caller(self, path: str, method_name: str, **kwargs) -> Any:
        """
        Calls any method of the Storage that was defined in the tensor definition

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        method_name: str
            Name of the method used by the Storage

        Returns
        -------
        The result vary depending on the method called
        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': method_name}})

    def read(self, path: str, **kwargs) -> xarray.DataArray:
        """
        Calls the read method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choose.

        Returns
        -------
        An xarray.DataArray that allow to read the data in the path.

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'read'}})

    def append(self, path: str, **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        Calls the append method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.
        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Returns a list of xarray.backends.common.AbstractWritableDataStore objects,
        which is a class used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'append'}})

    def update(self, path: str, **kwargs) -> xarray.backends.common.AbstractWritableDataStore:
        """
        Calls the update method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.
        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Returns a single the xarray.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'update'}})

    def store(self, path: str, **kwargs) -> xarray.backends.common.AbstractWritableDataStore:
        """
        Calls the store method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Returns a single the xarray.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'store'}})

    def upsert(self, path: str, **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        Calls the upsert method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Returns a List of xarray.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'upsert'}})

    def backup(self, path: str, **kwargs) -> xarray.DataArray:
        """
        Calls the backup method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Returns a List of xarray.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).
        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'backup'}})

    def update_from_backup(self, path: str, **kwargs) -> Any:
        """
        Calls the update_from_backup method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        Dependens of the Storage used.
        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'update_from_backup'}})

    def set_attrs(self, path: str, **kwargs):
        """
        Calls the set_attrs method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'set_attrs'}})

    def get_attrs(self, path: str, **kwargs) -> Dict:
        """
        Calls the update_from_backup method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        A dict with the attributes of the tensor (metadata).
        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'get_attrs'}})

    def close(self, path: str, **kwargs) -> xarray.DataArray:
        """
        Calls the close method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.
        **kwargs
            Parameters used for the internal Storage that you choosed.
        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'close'}})

    def delete_file(self, path: str, **kwargs) -> Any:
        """
        Calls the delete_file method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        """
        return self._customize_handler_action(path=path, **{**kwargs, **{'action_type': 'delete_file'}})

    def exist(self, path: str, **kwargs) -> bool:
        """
        Calls the exist method of the corresponding Storage defined for the tensor in the path
        (read `BaseStorage` for more info of this method or read the specific doc of your Storage).

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        A bool indicating if the file exist or not (True means yes).
        """
        # TODO: this method fail if the tensor was not created, so this must be fixed it should return False
        return self._get_handler(path).exist(**kwargs)

    def exist_tensor_definition(self, path: str):
        """
        Check if exist an specific definition.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        Returns
        -------
        A bool indicating if the definition exist or not (True means yes).

        """
        base_storage = BaseStorage(path, self.local_base_map, self.backup_base_map)
        return 'tensor_definition.json' in base_storage.backup_map

    def get_cached_tensor_manager(self, path, max_cached_in_dim: int, dim: str, **kwargs):
        """
        Create a `CachedTensorHandler` object which is used for multiples writes of the same file.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        max_cached_in_dim: int
            ``CachedTensorHandler.max_cached_in_dim``

        dim: str
            ``CachedTensorHandler.dim``

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        A `CachedTensorHandler` object.

        """
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
        """
        This is one of the most important methods of the ``TensorClient`` class, basically it allows to define
        formulas that use the tensors stored with a simple strings, so you can create new tensors from this formulas
        (make use of python eval). This is very flexible and the only thing you need to know is that you have
        to wrap the path of your tensor with "`" to be parsed and read it automatically.

        Another important chracteristic is that you can even pass entiere python codes to create this new tensors
        (it make use of python exec so use use_exec as True).

        Parameters
        ----------
        tensor_definition: Dict, optional
            Definition of your tensor.

        new_data: xarray.DataArray, optional
            Sometimes you can use this method in combination with others so you can pass the data that you are
            creating using this parameters (is more for internal use).

        use_exec: bool = False
            Indicate if you want to use python exec or eval for the formula.

        **kwargs
            Extra parameters used principally for when you want to use the exec option and want to add some settings
            or values.

        Examples
        --------
        You can see some examples of this method in the example of TensorClient, I will put more examples
        here in the future

        Returns
        -------
        An xarray.DataArray object created from the formula.

        """
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
