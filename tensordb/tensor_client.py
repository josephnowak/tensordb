import xarray as xr
import numpy as np
import dask

from typing import Dict, List, Any, Union, Tuple, Literal
from collections.abc import MutableMapping

from pandas import Timestamp
from loguru import logger
from pydantic import validate_arguments

from tensordb.storages import (
    BaseStorage,
    JsonStorage,
    CachedStorage,
    MAPPING_STORAGES
)
from tensordb.utils.method_inspector import get_parameters
from tensordb import algorithms
from tensordb.tensor_definition import TensorDefinition, MethodDescriptor, Definition
from tensordb import dag


class TensorClient:

    """

    The client was designed to handle multiple tensors data in a simpler way using Xarray in the background,
    it can support the same files than Xarray but those formats needs to be implemented
    using the `BaseStorage` interface proposed in this package.

    As we can create Tensors with multiple Storage that needs differents settings or parameters we must create
    a "Tensor Definition" which is basically a json that specify the behaviour of the tensor
    every time you call a method of one Storage. The definitions are very simple to create (there are few internal keys)
    , you only need to use as key the name of your method and as value a dictionary containing all the necessary
    parameters to use it, you can see some examples in the ``Examples`` section

    Additional features:
        1. Support for any file system that implements the MutableMapping interface.
        2. Creation or modification of new tensors using dynamic string formulas (even python code (string)).
        3. The read method return a lazy Xarray DataArray or Dataset instead of only retrieve the data.
        4. It's easy to inherit the class and add customized methods.
        5. You can use any storage supported by the Zarr protocole to store your data using the ZarrStorage class,
           so you don't have to always use files, you can even store the tensors in
           `MongoDB <https://zarr.readthedocs.io/en/stable/api/storage.html.>`_


    Parameters
    ----------
    base_map: fsspec.FSMap (or any File System that implement the MutabbleMapping Interface)
       File system to store all tensors.

    synchronizer: str
        Some of the Storages used to handle the files support a synchronizer, this parameter is used as a default
        synchronizer option for everyone of them (you can pass different synchronizer to every tensor).

    **kwargs: Dict
        Useful when you want to inherent from this class.

    Examples
    --------

    Store and read a simple tensor:

        >>> import xarray as xr
        >>> import fsspec
        >>> from tensordb import TensorClient, tensor_definition
        >>>
        >>> tensor_client = TensorClient(
        ...     base_map=fsspec.get_mapper('test_db'),
        ...     synchronizer='thread'
        ... )
        >>>
        >>> # create a new empty tensor, you must always call this method to start using the tensor.
        >>> tensor_client.create_tensor(
        ...     tensor_definition.TensorDefinition(
        ...         path='tensor1',
        ...         definition={},
        ...         storage={
        ...             # modify the default Storage for the zarr_storage
        ...             'storage_name': 'zarr_storage'
        ...         }
        ...     )
        ... )
        >>>
        >>> new_data = xr.DataArray(
        ...     0.0,
        ...     coords={'index': list(range(3)), 'columns': list(range(3))},
        ...     dims=['index', 'columns']
        ... )
        >>>
        >>> # Storing tensor1 on disk
        >>> tensor_client.store(path='tensor1', new_data=new_data)
        <xarray.backends.zarr.ZarrStore object at 0x0000023441FFAE40>
        >>>
        >>> # Reading the tensor1 (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read(path='tensor1').compute()
        <xarray.DataArray 'data' (index: 3, columns: 3)>
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        Coordinates:
          * columns  (columns) int32 0 1 2
          * index    (index) int32 0 1 2


    Storing a tensor from a string formula (if you want to create an 'on the fly' tensor using formula see
    the docs :meth:`TensorClient.read_from_formula`:

        >>> # create a new tensor using a formula that depend on the previous stored tensor
        >>> # Note: The tensor is not calculated or stored when you create the tensor
        >>> tensor_client.create_tensor(
        ...     tensor_definition.TensorDefinition(
        ...         path='tensor_formula',
        ...         definition={
        ...             'store': {
        ...                 # read the docs `TensorDefinition` for more info about the data_transformation
        ...                 'data_transformation': [
        ...                     {'method_name': 'read_from_formula'}
        ...                 ],
        ...             },
        ...             'read_from_formula': {
        ...                 'formula': '`tensor1` + 1 + `tensor1` * 10'
        ...             }
        ...         }
        ...     )
        ... )
        >>>
        >>> # Storing tensor_formula on disk, check that now we do not need to send the new_data parameter, because it is generated
        >>> # from the formula that we create previously
        >>> tensor_client.store(path='tensor_formula')
        <xarray.backends.zarr.ZarrStore object at 0x000002061203B200>
        >>>
        >>> # Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read(path='tensor_formula').compute()
        <xarray.DataArray 'data' (index: 3, columns: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * columns  (columns) int32 0 1 2
          * index    (index) int32 0 1 2

    Appending a new row and a new column to a tensor:

        >>>
        >>> # Appending a new row and a new columns to the tensor_formula stored previously
        >>> new_data = xr.DataArray(
        ...     2.,
        ...     coords={'index': [3], 'columns': list(range(4))},
        ...     dims=['index', 'columns']
        ... )
        >>>
        >>> # Appending the data, you can use the compute=False parameter if you dont want to execute this immediately
        >>> tensor_client.append('tensor_formula', new_data=new_data)
        [<xarray.backends.zarr.ZarrStore object at 0x000002061203BE40>,
        <xarray.backends.zarr.ZarrStore object at 0x000002061203BB30>]
        >>>
        >>> # Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
        >>> tensor_client.read('tensor_formula').compute()
        <xarray.DataArray 'data' (index: 4, columns: 4)>
        array([[ 1.,  1.,  1., nan],
               [ 1.,  1.,  1., nan],
               [ 1.,  1.,  1., nan],
               [ 2.,  2.,  2.,  2.]])
        Coordinates:
          * columns  (columns) int32 0 1 2 3
          * index    (index) int32 0 1 2 3
    """

    # TODO: Add more examples to the documentation

    internal_actions = ['store', 'update', 'append', 'upsert']

    def __init__(self,
                 base_map: MutableMapping,
                 synchronizer: str = None,
                 **kwargs):

        self.base_map = base_map
        self.synchronizer = synchronizer
        self._tensors_definition = JsonStorage(
            path='_tensors_definition',
            base_map=self.base_map,
        )

    @validate_arguments
    def create_tensor(self, definition: TensorDefinition):
        """
        Store the definition of tensor, which is equivalent to the creation of the tensor but without data

        Parameters
        ----------
        definition: TensorDefinition
            Read the docs of the `TensorDefinition` class for more info of the definition.

        """
        self._tensors_definition.store(new_data=definition.dict(exclude_unset=True), name=definition.path)

    @validate_arguments
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        """
        Retrieve the tensor definition of an specific tensor.

        Parameters
        ----------
        path: str
            Location of your stored tensor.

        Returns
        -------
        A `TensorDefinition`

        """
        try:
            return TensorDefinition(**self._tensors_definition.read(path))
        except KeyError:
            raise KeyError('You can not use a tensor without first call the create_tensor method')

    def get_all_tensors_definition(self) -> List[TensorDefinition]:
        return [self.get_tensor_definition(path) for path in self._tensors_definition.base_map.keys()]

    @validate_arguments
    def delete_tensor(self, path: str, only_data: bool = False) -> Any:
        """
        Delete the tensor

        Parameters
        ----------

        only_data: bool, default False
            If this option is marked as True only the data will be erased and not the definition

        """

        storage = self.get_storage(path)
        storage.base_map.clear()
        self.base_map.fs.rmdir(storage.base_map.root)
        if not only_data:
            self._tensors_definition.delete_file(path)

    @validate_arguments
    def get_storage(self, path: Union[str, TensorDefinition]) -> BaseStorage:
        """
        Get the storage of the tensor, by default it try to read the stored definition of the tensor.

        Parameters
        ----------
        path: Union[str, TensorDefinition]
            Location of your stored tensor or a `TensorDefinition`

        Returns
        -------
        A BaseStorage object
        """
        definition = self.get_tensor_definition(path) if isinstance(path, str) else path

        storage = MAPPING_STORAGES['zarr_storage']

        if definition.storage.synchronizer is None:
            definition.storage.synchronizer = self.synchronizer

        storage = MAPPING_STORAGES[definition.storage.storage_name]
        storage = storage(
            base_map=self.base_map,
            path=definition.path,
            **definition.storage.dict(exclude_unset=True)
        )
        return storage

    def exec_on_dag_order(
            self,
            method: Literal['append', 'update', 'store', 'upsert'],
            kwargs_groups: Dict[str, Dict[str, Any]] = None,
            tensors_path: List[str] = None,
            client: dask.distributed.Client = None,
    ):
        kwargs_groups = {} if kwargs_groups is None else kwargs_groups
        if tensors_path is None:
            tensors = [tensor for tensor in self.get_all_tensors_definition() if tensor.dag is not None]
        else:
            tensors = [self.get_tensor_definition(path) for path in tensors_path]

        for level in dag.get_tensor_dag(tensors):
            futures = [
                getattr(self, method)(
                    path=tensor.path,
                    compute=False,
                    **kwargs_groups.get(tensor.dag.group, {})
                )
                for tensor in level
            ]
            if client is None:
                dask.compute(*futures, sync=True)
            else:
                client.compute(futures, sync=True)

    def apply_data_transformation(
            self,
            data_transformation: List[MethodDescriptor],
            storage: BaseStorage,
            definition: Dict[str, Definition],
            parameters: Dict[str, Any]
    ):
        parameters = {**{'new_data': None}, **parameters}
        for descriptor in data_transformation:
            func = getattr(storage, descriptor.method_name, None)
            if func is None:
                func = getattr(self, descriptor.method_name)

            method_parameters = descriptor.parameters
            if descriptor.method_name in definition:
                method_parameters = {**method_parameters, **definition[descriptor.method_name].dict(exclude_unset=True)}
            result = func(**get_parameters(func, method_parameters, parameters))

            if descriptor.result_name is not None:
                parameters.update({descriptor.result_name: result})
            else:
                parameters.update(result if isinstance(result, dict) else {'new_data': result})

        return parameters['new_data']

    @validate_arguments
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

    @validate_arguments
    def storage_method_caller(
            self,
            path: Union[str, TensorDefinition],
            method_name: str,
            parameters: Dict[str, Any]
    ) -> Any:
        """
        Calls an specific method of a Storage, this include send the parameters specified in the tensor_definition
        or modifying the behaviour of the method based in your tensor_definition
        (read :meth:`TensorDefinition` for more info of how to personalize your method).

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
        tensor_definition = self.get_tensor_definition(path) if isinstance(path, str) else path
        definition = tensor_definition.definition

        parameters.update({'definition': definition})
        storage = self.get_storage(path=tensor_definition)

        if method_name in definition:
            method_settings = definition[method_name]

            if method_settings.substitute_method is not None:
                func = getattr(self, method_settings.substitute_method)
                if func.__name__ in self.internal_actions:
                    return func(path=path, **parameters)
                return func(**get_parameters(
                    func,
                    parameters,
                    definition[func.__name__].dict(exclude_unset=True) if func.__name__ in definition else {}
                ))

            if method_settings.data_transformation is not None:
                parameters['new_data'] = self.apply_data_transformation(
                    data_transformation=method_settings.data_transformation,
                    storage=storage,
                    definition=definition,
                    parameters=parameters
                )

        func = getattr(storage, method_name)
        return func(**get_parameters(func, parameters))

    def read(self, path: str, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with read as method_name (has the same parameters).

        Returns
        -------
        An xr.DataArray that allow to read the data in the path.

        """
        return self.storage_method_caller(path=path, method_name='read', parameters=kwargs)

    def append(self, path: str, **kwargs) -> List[xr.backends.common.AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with append as method_name (has the same parameters).

        Returns
        -------
        Returns a List of xr.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='append', parameters=kwargs)

    def update(self, path: str, **kwargs) -> xr.backends.common.AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with update as method_name (has the same parameters).

        Returns
        -------
        Returns a single the xr.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='update', parameters=kwargs)

    def store(self, path: str, **kwargs) -> xr.backends.common.AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with store as method_name (has the same parameters).

        Returns
        -------
        Returns a single the xr.backends.common.AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='store', parameters=kwargs)

    def upsert(self, path: str, **kwargs) -> List[xr.backends.common.AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with upsert as method_name (has the same parameters).

        Returns
        -------
        Returns a List of xr.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='upsert', parameters=kwargs)

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
            new_data: xr.DataArray = None,
            use_exec: bool = False,
            **kwargs
    ) -> xr.DataArray:
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
        new_data: xr.DataArray, optional
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
            >>> import fsspec
            >>> from tensordb import TensorClient
            >>> from tensordb import tensor_definition
            >>>
            >>> tensor_client = TensorClient(
            ...     base_map=fsspec.get_mapper('test_db'),
            ...     synchronizer='thread'
            ... )
            >>> # Creating a new tensor definition using an 'on the fly' formula
            >>> tensor_client.create_tensor(
            ...     tensor_definition.TensorDefinition(
            ...         path='tensor_formula_on_the_fly',
            ...         definition={
            ...             'read': {
            ...                 # Read the section reserved Keywords
            ...                 'substitute_method': 'read_from_formula',
            ...             },
            ...             'read_from_formula': {
            ...                 'formula': '`tensor1` + 1 + `tensor1` * 10'
            ...             }
            ...         }
            ...     )
            ... )
            >>>
            >>> # Now we don't need to call the store method when we want to read our tensor
            >>> # the good part is that everything is still lazy
            >>> tensor_client.read(path='tensor_formula_on_the_fly').compute()
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
        An xr.DataArray object created from the formula.

        """

        data_fields = {}
        data_fields_intervals = np.array([i for i, c in enumerate(formula) if c == '`'])
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

    @classmethod
    def ffill(
            cls,
            new_data: xr.DataArray,
            dim: str,
            limit: int = None,
            until_last_valid: Union[xr.DataArray, bool] = False,
            keep_chunks_size: bool = False
    ):
        return algorithms.ffill(
            arr=new_data,
            limit=limit,
            dim=dim,
            until_last_valid=until_last_valid,
            keep_chunks_size=keep_chunks_size
        )

    @classmethod
    def rank(
            cls,
            new_data: xr.DataArray,
            dim: str,
            method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'ordinal',
            rank_nan: bool = False
    ):
        return algorithms.rank(
            arr=new_data,
            method=method,
            dim=dim,
            rank_nan=rank_nan
        )

    @classmethod
    def shift_on_valids(
            cls,
            arr: xr.DataArray,
            dim: str,
            shift: int
    ):
        return algorithms.shift_on_valids(
            arr=new_data,
            dim=dim,
            shift=shift
        )