import xarray

from typing import Dict, List, Any, Union, Tuple, Literal
from numpy import nan, array
from pandas import Timestamp
from collections.abc import MutableMapping
from loguru import logger

from tensordb.storages import (
    BaseStorage,
    JsonStorage,
    CachedStorage,
    MAPPING_STORAGES
)
from tensordb.utils.method_inspector import get_parameters


class TensorClient:

    """

    The client was designed to handle multiple tensor data in a simpler way using Xarray in the background,
    it can support the same files than Xarray but those formats needs to be implemented
    using the `BaseStorage` interface proposed in this package.

    As we can create Tensors with multiple Storage that needs differents settings or parameters we must create
    a "Tensor Definition" which is basically a json that specify the behaviour of the tensor
    every time you call a method of one Storage. The definitions are very simple to create (there are few internal keys)
    , you only need to use as key the name of your method and as value a dictionary containing all the necessary
    parameters to use it, you can see some examples in the ``Examples`` section

    Additional features:
        1. Support for any backup system that implements the MutableMapping interface.
        2. Creation or modification of new tensors using dynamic string formulas (even python code (string)).
        3. The read method return a lazy Xarray DataArray instead of only retrieve the data.
        4. It's easy to inherit the class and add customized methods.
        5. The backups can be faster and saver because you can modify them as you want, an example of this is the
           ZarrStorage which has a checksum (currently only store a date for debug porpuse) of every chunk of every
           tensor stored to avoid uploading or downloading unnecessary data and is useful to check
           the integrity of the data.
        6. You can use any storage supported by the Zarr protocole to store your data using the ZarrStorage class,
           so you don't have to always use files, you can even store the tensors in
           `MongoDB <https://zarr.readthedocs.io/en/stable/api/storage.html.>`_


    Parameters
    ----------
    base_map: MutableMapping (normally fsspec.FSMap)
       MutableMapping instaciated with the path where you want to store all tensors.

    synchronizer: str
        Some of the Storages used to handle the files support a synchronizer, this parameter is used as a default
        synchronizer option for everyone of them (you can pass different synchronizer to every tensor).

    **kwargs: Dict
        Useful when you want to inherent from this class.

    Examples
    --------

    Store and read a simple tensor:

        >>> from tensordb import TensorClient
        >>> import xarray
        >>> import fsspec
        >>>
        >>> tensor_client = TensorClient(
        ...     local_base_map=fsspec.get_mapper('test_db'),
        ...     backup_base_map=fsspec.get_mapper('test_db' + '/backup'),
        ...     synchronizer='thread'
        ... )
        >>>
        >>> # Adding an empty tensor definition (there is no personalization)
        >>> tensor_client.add_definition(
        ...     definition_id='dummy_tensor_definition',
        ...     new_data={
        ...         # This key is used for modify options of the Storage constructor
        ...         # (documented on the reserved keys section of this method)
        ...         'handler': {
        ...             # modify the default Storage for the zarr_storage
        ...             'data_handler': 'zarr_storage'
        ...         }
        ...     }
        ... )
        >>>
        >>> # create a new empty tensor, you must always call this method to start using the tensor.
        >>> tensor_client.create_tensor(path='tensor1', definition='dummy_tensor_definition')
        >>>
        >>> new_data = xarray.DataArray(
        ...     0.0,
        ...     coords={'index': list(range(3)), 'columns': list(range(3))},
        ...     dims=['index', 'columns']
        ... )
        >>>
        >>> # Storing tensor1 on disk
        >>> tensor_client.store(path='tensor1', new_data=new_data)
        <xarray.backends.zarr.ZarrStore object at 0x000001FFBE9ADB80>
        >>>
        >>> # Reading the tensor1 (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read(path='tensor1')
        <xarray.DataArray 'data' (index: 3, columns: 3)>
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        Coordinates:
          * columns  (columns) int32 0 1 2
          * index    (index) int32 0 1 2

    Storing a tensor from a string formula (if you want to create an 'on the fly' tensor using formula see
    the docs :meth:`TensorClient.read_from_formula`:

        >>> # Creating a new tensor definition using a formula that depend on the previous stored tensor
        >>> tensor_client.add_definition(
        ...     definition_id='tensor_formula',
        ...     new_data={
        ...         'store': {
        ...             # read the docs of this method to understand the behaviour of the data_transformation key
        ...             'data_transformation': ['read_from_formula'],
        ...         },
        ...         'read_from_formula': {
        ...             'formula': '`tensor1` + 1 + `tensor1` * 10'
        ...         }
        ...     }
        ... )
        >>>
        >>> # create a new empty tensor, you must always call this method to start using the tensor.
        >>> tensor_client.create_tensor(path='tensor_formula', definition='tensor_formula')
        >>>
        >>> # Storing tensor_formula on disk, check that now we do not need to send the new_data parameter, because it is generated
        >>> # from the formula that we create previously
        >>> tensor_client.store(path='tensor_formula')
        <xarray.backends.zarr.ZarrStore object at 0x000001FFBEA93C40>
        >>>
        >>> # Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read(path='tensor_formula')
        <xarray.DataArray 'data' (index: 3, columns: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * columns  (columns) int32 0 1 2
          * index    (index) int32 0 1 2

    Appending a new row and a new column to a tensor:

        >>> # Appending a new row and a new columns to the tensor_formula stored previously
        >>> new_data = xarray.DataArray(
        ...     2.,
        ...     coords={'index': [3], 'columns': list(range(4))},
        ...     dims=['index', 'columns']
        ... )
        >>>
        >>> # Appending the data, you can use the compute=False parameter if you dont want to execute this immediately
        >>> tensor_client.append('tensor_formula', new_data=new_data)
        [<xarray.backends.zarr.ZarrStore object at 0x000001FFBEB77AC0>, <xarray.backends.zarr.ZarrStore object at 0x000001FFBEB779A0>]
        >>>
        >>> # Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read('tensor_formula')
        <xarray.DataArray 'data' (index: 4, columns: 4)>
        array([[ 1.,  1.,  1., nan],
               [ 1.,  1.,  1., nan],
               [ 1.,  1.,  1., nan],
               [ 2.,  2.,  2.,  2.]])
        Coordinates:
          * columns  (columns) int32 0 1 2 3
          * index    (index) int32 0 1 2 3

    TODO:
        1. Add more examples to the documentation

    """

    internal_actions = ['store', 'update', 'append', 'upsert']

    def __init__(self,
                 base_map: MutableMapping,
                 synchronizer: str = None,
                 **kwargs):

        self.base_map = base_map

        self.open_base_store: Dict[str, Dict[str, Any]] = {}
        self.synchronizer = synchronizer
        self._definitions = JsonStorage(
            path='tensor_client/definitions',
            base_map=self.base_map,
        )
        self._tensors_definition = JsonStorage(
            path='tensor_client/tensors_definition',
            base_map=self.base_map,
        )

    def add_definition(self, definition_id: str, new_data: Dict) -> Dict:
        """
        Add (store) a new definition for a tensor (internally is stored as a JSON file).

        Reserved Keywords:
            storage: This key is used to personalize the Storage used for the tensor,
            inside it you can use the next reserved keywords:

                1. storage_name: Here you put the name of your storage (default zarr_storage), you can see
                all the names in the variable MAPPING_STORAGES.

            You can personalize the way that any Storage method is used specifying it in the tensor_definition,
            this is basically add a key with the name of the method and inside of it you can add any kind of parameters,
            but there are some reserved words that are used by the Tensorclient to add specific functionalities,
            these are described here:

                1. data_transformation:
                    Receive a list where every position can be a name of a method of the client or the storage or
                    a list containing two elements, the first is the name of the method and the second the parameters
                    of the method that you want to sent

                2. substitute_method:
                    Modify the method called, this is useful if you want to overwrite the defaults
                    methods of storage, read, etc for some specific tensors, this is normally used when you want to read
                    a tensor on the fly with a formula.


        Parameters
        ----------
        definition_id: str
            name used to identify the tensor definition.

        new_data: Dict
            Description of the definition, find an example of the format here.

        Examples
        --------
        Add examples.

        See Also:
            Read the :meth:`TensorClient.storage_method_caller` to learn how to personalize your methods

        """
        self._definitions.store(name=definition_id, new_data=new_data)

    def create_tensor(self, path: str, definition: Union[str, Dict]):
        """
        Create the path and the first file of the tensor which store the necessary metadata to use it,
        this method must always be called before start to write in the tensor.

        Parameters
        ----------
        path: str
            Indicate the location where your tensor is going to be allocated.

        definition: str, Dict
            This can be an string which allow to read a previously created tensor definition or a completly new
            tensor_definition in case that you pass a Dict.

        See Also:
            If you want to personalize any method of your Storage read the `TensorClient.storage_method_caller` doc


        """
        self._tensors_definition.store(new_data={'definition': definition}, name=path)

    def get_definition(self, name: str) -> Dict:
        """
        Retrieve a definition.

        Parameters
        ----------
        name: str
            name of the tensor definition.

        Returns
        -------
        A dict containing all the information of the tensor definition previusly stored.

        """
        return self._definitions.read(name)

    def get_tensor_definition(self, path) -> Dict:
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
        try:
            tensor_definition = self._tensors_definition.read(path)['definition']
        except KeyError:
            raise KeyError('You can not use a tensor without first call the create_tensor method')
        if isinstance(tensor_definition, dict):
            return tensor_definition
        return self.get_definition(tensor_definition)

    def get_storage(self, path: str, definition: Dict = None) -> BaseStorage:
        """
        Get the storage of the tensor, by default it try to read the stored definition of the tensor.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        definition: Optiona[Dict]
            Definition for instance the storage, by default this method try to get the definition previously stored
            for the tensor with the create_tensor method

        Returns
        -------
        A BaseStorage object
        """
        storage_settings = self.get_tensor_definition(path) if definition is None else definition
        storage_settings = storage_settings.get('storage', {})
        storage_settings['synchronizer'] = storage_settings.get('synchronizer', self.synchronizer)

        storage = MAPPING_STORAGES[storage_settings.get('storage_name', 'zarr_storage')]
        storage = storage(
            base_map=self.base_map,
            path=path,
            **storage_settings
        )
        if path not in self.open_base_store:
            self.open_base_store[path] = {
                'first_read_date': Timestamp.now(),
                'num_use': 0
            }
        self.open_base_store[path]['storage'] = storage
        self.open_base_store[path]['num_use'] += 1
        return self.open_base_store[path]['storage']

    def customize_storage_method(self, path: str, method_name: str, parameters: Dict[str, Any]):
        definition = self.get_tensor_definition(path)
        method_settings = definition.get(method_name, {})
        parameters.update({'definition': definition})
        if 'substitute_method' in method_settings:
            func = getattr(self, method_settings['substitute_method'])
            if func.__name__ in self.internal_actions:
                return func(path=path, **parameters)
            return func(**get_parameters(func, parameters, definition.get(func.__name__, {})))

        storage = self.get_storage(path=path, definition=definition)
        if 'data_transformation' in method_settings:
            parameters['new_data'] = self.apply_data_transformation(
                data_transformation=method_settings['data_transformation'],
                storage=storage,
                definition=definition,
                parameters=parameters
            )

        func = getattr(storage, method_name)
        return func(**get_parameters(func, parameters))

    def apply_data_transformation(
            self,
            data_transformation: List[Union[str, Tuple[str, Dict]]],
            storage: BaseStorage,
            definition: Dict,
            parameters: Dict[str, Any]
    ):
        parameters = {**{'new_data': None}, **parameters}
        for method in data_transformation:
            if isinstance(method, (list, tuple)):
                method_name, method_parameters = method[0], method[1]
            else:
                method_name, method_parameters = method, definition.get(method, {})

            func = getattr(storage, method_name, None)
            if func is None:
                func = getattr(self, method_name)

            result = func(**get_parameters(func, method_parameters, parameters))
            default_result_name = method_parameters.get('result_name', 'new_data')
            parameters.update(result if isinstance(result, dict) else {default_result_name: result})

        return parameters['new_data']

    def storage_method_caller(self, path: str, method_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Calls an specific method of a Storage, this include send the parameters specified in the tensor_definition
        or modifying the behaviour of the method based in your tensor_definition
        (read :meth:`TensorClient.add_definition` for more info of how to personalize your method).

        If you want to know the specific behaviour of the method that you are using,
        please read the specific documentation of the Storage that you are using or read `BaseStorage` .

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        method_name: str
            Name of the method used by the Storage.

        parameters: Dict
            Parameters that are going to be used by the Storage method, in case that any of this parameter
            match with the ones provided in the tensor_definition they will overwrite them.

        Returns
        -------
        The result vary depending on the method called.

        """
        return self.customize_storage_method(path=path, method_name=method_name, parameters=parameters)

    def read(self, path: str, **kwargs) -> Union[xarray.DataArray, xarray.Dataset]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with read as method_name (has the same parameters).

        Returns
        -------
        An xarray.DataArray that allow to read the data in the path.

        """
        return self.storage_method_caller(path=path, method_name='read', parameters=kwargs)

    def append(self, path: str, **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with append as method_name (has the same parameters).

        Returns
        -------
        Returns a List of xarray.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='append', parameters=kwargs)

    def update(self, path: str, **kwargs) -> xarray.backends.common.AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with update as method_name (has the same parameters).

        Returns
        -------
        Returns a single the xarray.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='update', parameters=kwargs)

    def store(self, path: str, **kwargs) -> xarray.backends.common.AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with store as method_name (has the same parameters).

        Returns
        -------
        Returns a single the xarray.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='store', parameters=kwargs)

    def upsert(self, path: str, **kwargs) -> List[xarray.backends.common.AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with upsert as method_name (has the same parameters).

        Returns
        -------
        Returns a List of xarray.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='upsert', parameters=kwargs)

    def delete_tensor(self, path: str) -> Any:
        """
        Delete a tensor
        """

        if self._tensors_definition.exist(path):
            storage = self.get_storage(path)
            storage.base_map.clear()
            self.base_map.fs.rmdir(storage.base_map.root)
            self._tensors_definition.delete_file(path)

    def delete_definition(self, definition_id: str, delete_tensors: bool):
        """
        Delete a definition and all the tensor that use the definition (only if delete_tensors = True)

        Parameters
        ----------
        definition_id: str
            name used to identify the tensor definition.

        delete_tensors: bool
            True means delete all the tensors that use the definition, False means don't delete them

        """
        if self._definitions.exist(definition_id):
            if delete_tensors:
                for name in self._tensors_definition.base_map.keys():
                    original_path = self._tensors_definition.get_original_path(name)
                    tensor_definition = self._tensors_definition.read(original_path)['definition']
                    if not isinstance(tensor_definition, str):
                        continue
                    if tensor_definition == definition_id:
                        self.delete_tensor(original_path)
            self._definitions.delete_file(definition_id)

    def exist(self, path: str, **kwargs) -> bool:
        """
        Calls :meth:`TensorClient.storage_method_caller` with exist as method_name (has the same parameters).

        Returns
        -------
        A bool indicating if the file exist or not (True means yes).
        """
        try:
            return self._tensors_definition.exist(path) and self.get_storage(path).exist(**kwargs)
        except KeyError:
            return False

    def get_cached_storage(self, path, max_cached_in_dim: int, dim: str, **kwargs):
        """
        Create a `CachedStorage` object which is used for multiples writes of the same file.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        max_cached_in_dim: int
            `CachedStorage.max_cached_in_dim`

        dim: str
            `CachedStorage.dim`

        **kwargs
            Parameters used for the internal Storage that you choosed.

        Returns
        -------
        A `CachedStorage` object.

        """
        return CachedStorage(
            storage=self.get_storage(path, **kwargs),
            max_cached_in_dim=max_cached_in_dim,
            dim=dim
        )

    def read_from_formula(
            self,
            formula: str,
            new_data: xarray.DataArray = None,
            use_exec: bool = False,
            **kwargs
    ) -> xarray.DataArray:
        """
        This is one of the most important methods of the `TensorClient` class, basically it allows to define
        formulas that use the tensors stored with a simple strings, so you can create new tensors from this formulas
        (make use of python eval and the same syntax that you use with Xarray).
        This is very flexible, you can even create relations between tensor and the only extra thing
        you need to know is that you have to wrap the path of your tensor with "`" to be parsed and
        read automatically.

        Another important chracteristic is that you can even pass entiere python codes to create this new tensors
        (it make use of python exec so use use_exec parameter as True).

        Parameters
        ----------
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

        Reading a tensor directly from a formula, all this is lazy evaluated:
            >>> # Creating a new tensor definition using an 'on the fly' formula
            >>> tensor_client.add_definition(
            ...     definition_id='tensor_formula_on_the_fly',
            ...     new_data={
            ...         'read': {
            ...             # Read the section reserved Keywords
            ...             'substitute_method': 'read_from_formula',
            ...         },
            ...         'read_from_formula': {
            ...             'formula': '`tensor1` + 1 + `tensor1` * 10'
            ...         }
            ...     }
            ... )
            >>>
            >>> # create a new empty tensor, you must always call this method to start using the tensor.
            >>> tensor_client.create_tensor(path='tensor_formula_on_the_fly', definition='tensor_formula_on_the_fly')
            >>>
            >>> # Now we don't need to call the store method when we want to read our tensor
            >>> # the good part is that everything is still lazy
            >>> tensor_client.read(path='tensor_formula_on_the_fly')
            <xarray.DataArray 'data' (index: 3, columns: 3)>
            array([[1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.]])
            Coordinates:
              * columns  (columns) int32 0 1 2
              * index    (index) int32 0 1 2

        You can see an example of how to store a tensor from a formula in the examples of the
        constructor section in `TensorClient`

        Returns
        -------
        An xarray.DataArray object created from the formula.

        """

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

    def append_reindex(
            self,
            new_data: xarray.DataArray,
            reindex_data: Union[str, xarray.DataArray],
            dims_to_reindex: List[str],
            method_fill_value: str = None
    ) -> Union[xarray.DataArray, None]:
        if isinstance(reindex_data, str):
            reindex_data = self.read(path=reindex_data)
        coords_to_reindex = {
            coord: reindex_data.coords[coord][reindex_data.coords[coord] >= new_data.coords[coord][-1].values]
            for coord in dims_to_reindex
        }
        return new_data.reindex(coords_to_reindex, method=method_fill_value)

    def reindex(
            self,
            new_data: xarray.DataArray,
            reindex_data: Union[str, xarray.DataArray],
            dims_to_reindex: List[str],
            method_fill_value: str = None
    ) -> Union[xarray.DataArray, None]:
        if isinstance(reindex_data, str):
            reindex_data = self.read(path=reindex_data)
        coords_to_reindex = {dim: reindex_data.coords[dim] for dim in dims_to_reindex}
        return new_data.reindex(coords_to_reindex, method=method_fill_value)

    def last_valid_dim(
            self,
            new_data: xarray.DataArray,
            dim: str
    ) -> Union[xarray.DataArray, None]:
        if new_data.dtype == 'bool':
            return new_data.cumsum(dim=dim).idxmax(dim=dim)
        return new_data.notnull().cumsum(dim=dim).idxmax(dim=dim)

    def replace_by_last_valid(
            self,
            new_data: xarray.DataArray,
            last_valid_data: Union[str, xarray.DataArray],
            dim: str,
            value: Any = nan,
            replace_method: Literal['after', 'before'] = 'after',
    ) -> Union[xarray.DataArray, None]:
        if isinstance(last_valid_data, str):
            last_valid_data = self.read(last_valid_data)
        if replace_method == 'after':
            last_valid_data = new_data.coords[dim] <= last_valid_data.fillna(new_data.coords[dim][-1])
        else:
            last_valid_data = new_data.coords[dim] >= last_valid_data.fillna(new_data.coords[dim][-1])
        return new_data.where(last_valid_data.sel(new_data.coords), value)

    def replace_where(
            self,
            new_data: xarray.DataArray,
            bitmask: Union[str, xarray.DataArray],
            value: Any = nan
    ) -> Union[xarray.DataArray, None]:
        if isinstance(bitmask, str):
            bitmask = self.read(bitmask)
        return new_data.where(bitmask.sel(new_data.coords), value)

    def fillna(
            self,
            new_data: xarray.DataArray,
            value: Any
    ) -> Union[xarray.DataArray, None]:
        return new_data.fillna(value)

    def ffill(
            self,
            new_data: xarray.DataArray,
            dim: str,
            limit: int = None,
    ):
        return new_data.ffill(dim=dim, limit=limit)

    def append_ffill(
            self,
            storage: BaseStorage,
            new_data: xarray.DataArray,
            dim: str,
            limit: int = None,
            remote: bool = False,
    ) -> Union[xarray.DataArray, None]:

        data_concat = new_data
        act_data = storage.read(remote=remote)
        data = data.sel({dim: data.coords[dim] < new_data.coords[dim][0]})
        if data.sizes[dim] > 0:
            data_concat = xarray.concat([data.isel({dim: [-1]}), new_data], dim=dim)

        return data_concat.ffill(dim=dim, limit=limit).sel(new_data.coords)
