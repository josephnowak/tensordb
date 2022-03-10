import fsspec
import xarray as xr
import numpy as np
import pandas as pd
import dask
import dask.array as da

from typing import Dict, List, Any, Union, Tuple, Literal, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections.abc import MutableMapping

from loguru import logger
from pydantic import validate_arguments
from fsspec.implementations.cached import CachingFileSystem

from tensordb.storages import (
    BaseStorage,
    JsonStorage,
    CachedStorage,
    MAPPING_STORAGES
)
from tensordb.utils.method_inspector import get_parameters
from tensordb.algorithms import Algorithms
from tensordb.tensor_definition import TensorDefinition, MethodDescriptor, Definition, DAGOrder
from tensordb import dag


class TensorClient(Algorithms):

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

    tmp_map: fsspec.FSMap (or any File System that implement the MutabbleMapping Interface)
        File system for storing the temporal cache local file and also the rewrites of the tensors,
        the cache local file are stored on the path {tmp_map.root}/_local_cache_file/tensor_path

    local_cache_protocol: Literal['simplecache', 'filecache', 'cached']
        Indicate the kind of local cache filesystem that want to be use for all the tensors,
        Read fsspec fileprotocol for more info about the local cache protocols.

    local_cache_options: Dict[str, Any]
        Parameters of the local cache filesystem

    synchronizer: str
        Some Storages used to handle the files support a synchronizer, this parameter is used as a default
        synchronizer option for every one of them (you can pass different synchronizer to every tensor).

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

    internal_actions = ['store', 'update', 'append', 'upsert', 'drop']

    def __init__(
            self,
            base_map: fsspec.FSMap,
            tmp_map: fsspec.FSMap = None,
            local_cache_protocol: Literal['simplecache', 'filecache', 'cached'] = None,
            local_cache_options: Dict[str, Any] = None,
            synchronizer: str = None,
            **kwargs
    ):
        if isinstance(base_map.fs, CachingFileSystem):
            raise ValueError(
                f'TensorDB do not support directly a cache file system, use the local_cache_protocol parameter'
            )

        self.base_map = base_map
        self.tmp_map = fsspec.get_mapper('tmp') if tmp_map is None else tmp_map
        self.local_cache_protocol = local_cache_protocol
        self.local_cache_options = local_cache_options
        self.synchronizer = synchronizer
        self._tensors_definition = JsonStorage(
            path='_tensors_definition',
            base_map=self.base_map,
            tmp_map=self.tmp_map,
            # local_cache_protocol=self.local_cache_protocol
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
        path: str
            path of the tensor

        only_data: bool, default False
            If this option is marked as True only the data will be erased and not the definition

        """

        storage = self.get_storage(path)
        storage.delete_tensor()
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
            tmp_map=self.tmp_map,
            path=definition.path,
            local_cache_protocol=self.local_cache_protocol,
            local_cache_options=self.local_cache_options,
            **definition.storage.dict(exclude_unset=True)
        )
        return storage

    @classmethod
    def exec_on_parallel(
            cls,
            method: Callable,
            paths_kwargs: Dict[str, Dict[str, Any]],
            max_parallelization: int = None,
            compute: bool = False,
            client: dask.distributed.Client = None,
            call_pool: Literal['thread', 'process'] = 'thread',
            compute_kwargs: Dict[str, Any] = None,
    ):
        """
        This method was designed to execute multiple methods of the client on parallel

        Parameters
        ----------

        method: Callable
            method of the tensor client that is going to be executed on parallel

        paths_kwargs: Dict[str, Dict[str, Any]]
            The key represents the paths of the tensors and the values the kwargs for the method

        max_parallelization: int, default None
            Indicates the maximum number of parallel calls of the method, useful to restrict the parallelization
            and avoid overflow the server or any DB.
            None means call all in parallel

        compute: bool, default True
            Indicate if the computation must be delayed or not, read the use on zarr_storage doc for more info

        client: dask.distributed.Client, default None
            Dask client, useful for in combination with the compute parameter

        call_pool: Literal['thread', 'process'], default 'thread'
            Internally this method use a Pool of process/thread to apply the method on all the tensors
            this indicates the kind of pool

        compute_kwargs: Dict[str, Any], default None
            Parameters of the dask.compute or client.compute
        """
        paths = list(paths_kwargs.keys())

        call_pool = ThreadPoolExecutor if call_pool == 'thread' else ProcessPoolExecutor
        max_parallelization = np.inf if max_parallelization is None else max_parallelization
        max_parallelization = min(max_parallelization, len(paths))
        compute_kwargs = compute_kwargs or {}
        client = dask if client is None else client

        with call_pool(max_parallelization) as pool:
            for i in range(0, len(paths), max_parallelization):
                futures = [
                    pool.submit(
                        method,
                        path=path,
                        compute=compute,
                        **paths_kwargs[path]
                    )
                    for path in paths[i: i + max_parallelization]
                ]
                logger.info(
                    f'Waiting for the {"computed" if compute else "delayed"} '
                    f'execution of the method on all the tensor'
                )
                futures = [future.result() for future in futures]

                if not compute:
                    logger.info('Calling compute over all the delayed tensors')
                    client.compute(futures, **compute_kwargs)

    def exec_on_dag_order(
            self,
            method: Literal['append', 'update', 'store', 'upsert'],
            kwargs_groups: Dict[str, Dict[str, Any]] = None,
            tensors_path: List[str] = None,
            parallelization_kwargs: Dict[str, Any] = None,
            apply_on_dependencies: bool = False,
            only_on_groups: set = None,
    ):
        """
        This method was designed to execute multiple methods of the client on parallel and following the dag
        order, this is useful for update/store multiple tensors that have dependencies between them in a fast way.
        Internally calls the :meth:`TensorClient.exec_on_parallel` for every level of the created DAG

        Parameters
        ----------

        method: Literal['append', 'update', 'store', 'upsert']
            method of the tensor client that is going to be executed on parallel

        kwargs_groups: Dict[str, Dict[str, Any]]
            Kwargs sent to the method base on the DAG groups, read the docs of :meth:`DAGOrder` for more info

        tensors_path: List[str], default None
            Indicates the tensors on which the operation will be applied

        parallelization_kwargs: Dict[str, Any]
            Kwargs sent to :meth:`TensorClient.exec_on_parallel`

        apply_on_dependencies: bool, default False
            Not implemented, but is going to automatically fill the tensors_path

        only_on_groups: set, default None
            Useful for filters the tensors base on the DAG groups (read the DAG models of tensordb)

        """
        kwargs_groups = kwargs_groups or {}
        parallelization_kwargs = parallelization_kwargs or {}
        method = getattr(self, method)

        if tensors_path is None:
            tensors = [tensor for tensor in self.get_all_tensors_definition() if tensor.dag is not None]
        else:
            tensors = [self.get_tensor_definition(path) for path in tensors_path]
            if apply_on_dependencies:
                tensors = dag.get_dependencies(tensors)

        for level in dag.get_tensor_dag(tensors):
            # Filter the tensors base on the omit parameter
            level = [tensor for tensor in level if method.__name__ not in tensor.dag.omit_on]
            if only_on_groups:
                # filter the invalid groups
                level = [tensor for tensor in level if tensor.dag.group in only_on_groups]
            if not level:
                continue
            paths_kwargs = {
                tensor.path: kwargs_groups.get(tensor.dag.group, {})
                for tensor in level
            }
            self.exec_on_parallel(
                method=method,
                paths_kwargs=paths_kwargs,
                **parallelization_kwargs
            )

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
                method_parameters = {**definition[descriptor.method_name].dict(exclude_unset=True), **method_parameters}
            result = func(**get_parameters(func, parameters, method_parameters))

            if descriptor.result_name is not None:
                parameters.update({descriptor.result_name: result})
            else:
                parameters.update(result if isinstance(result, dict) else {'new_data': result})

        return parameters['new_data']

    @validate_arguments
    def storage_method_caller(
            self,
            path: Union[str, TensorDefinition],
            method_name: str,
            parameters: Dict[str, Any]
    ) -> Any:
        """
        Calls a specific method of a Storage, this includes send the parameters specified in the tensor_definition
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

        storage = self.get_storage(path=tensor_definition)
        parameters.update({
            'definition': definition,
            'original_path': path if isinstance(path, str) else path.path,
            'storage': storage
        })

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
                if method_name == 'read':
                    return parameters['new_data']

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

    def drop(self, path: str, **kwargs) -> List[xr.backends.common.AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with drop as method_name (has the same parameters).

        Returns
        -------
        Returns a List of xr.backends.common.AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='drop', parameters=kwargs)

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
            use_exec: bool = False,
            original_path: str = None,
            storage: BaseStorage = None,
            **kwargs: Dict[str, Any]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        This is one of the most important methods of the `TensorClient` class, basically it allows defining
        formulas that use the tensors stored with a simple strings, so you can create new tensors from these formulas
        (make use of python eval and the same syntax that you use with Xarray).
        This is very flexible, you can even create relations between tensor and the only extra thing
        you need to know is that you have to wrap the path of your tensor with "`" to be parsed and
        read automatically.

        Another important chracteristic is that you can even pass entiere python codes to create this new tensors
        (it makes use of python exec so use use_exec parameter as True).

        Note: The globals dictionary use in eval is the following:
        {'xr': xr, 'np': np, 'pd': pd, 'da': da, 'dask': dask, 'self': self}.
        You can use Pandas (pd), Numpy (np), Xarray (xr), Dask, Dask Array (da), self (client).

        Parameters
        ----------

        formula: str
            The formula is a string that is use in the python eval or exec function, this string is first analyzed
            to extract the tensor paths inside it to read them before execute the evaluation of the string,
            so, use the following syntax to indicate the part of the string that is a path of a tensor

        use_exec: bool = False
            Indicate if you want to use python exec or eval to evaluate the formula, it must always create
            a variable called new_data inside the code.

        original_path: str = None
            Useful to identify the original tensor path and avoid recursive reads

        storage: BaseStorage = None
            Storage of the actual tensor that is being read, this is the storage of the original_path
            useful for self reading, is normally send by the TensorClient class

        **kwargs: Dict
            It's use as the local dictionary of the eval and exec functions, it is automatically
            filled by the apply_data_transformation with all the previously stored data.

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
        An xr.DataArray or xr.Dataset object created from the formula.

        """

        data_fields = {}
        data_fields_intervals = np.array([i for i, c in enumerate(formula) if c == '`'])
        for i in range(0, len(data_fields_intervals), 2):
            name_data_field = formula[data_fields_intervals[i] + 1: data_fields_intervals[i + 1]]
            if original_path is not None and original_path == name_data_field:
                if storage is None:
                    raise ValueError(f'You can not make a self read without sending the storage parameter')
                data_fields[name_data_field] = storage.read()
            else:
                data_fields[name_data_field] = self.read(name_data_field)

        for name, dataset in data_fields.items():
            formula = formula.replace(f"`{name}`", f"data_fields['{name}']")

        formula_globals = {
            'xr': xr, 'np': np, 'pd': pd, 'da': da, 'dask': dask, 'self': self
        }
        kwargs.update({'data_fields': data_fields})

        if use_exec:
            exec(formula, formula_globals, kwargs)
            return kwargs['new_data']
        return eval(formula, formula_globals, kwargs)
