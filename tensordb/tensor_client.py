from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Union, Literal, Callable

import dask
import dask.array as da
import more_itertools as mit
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pydantic import validate_arguments
from tensordb import dag
from tensordb.algorithms import Algorithms
from tensordb.storages import (
    BaseStorage,
    JsonStorage,
    CachedStorage,
    MAPPING_STORAGES
)
from tensordb.storages.mapping import Mapping
from tensordb.tensor_definition import TensorDefinition, MethodDescriptor, Definition
from tensordb.utils.method_inspector import get_parameters
from tensordb.utils.tools import (
    groupby_chunks,
    extract_paths_from_formula,
    iter_by_group_chunks
)
from xarray.backends.common import AbstractWritableDataStore


class TensorClient(Algorithms):
    """

    The client was designed to handle multiple tensors' data in a simpler way using Xarray in the background,
    it can support the same files as Xarray but those formats needs to be implemented
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
    base_map: MutableMapping
       Mapping storage interface where all the tensors are stored.

    tmp_map: MutableMapping
        Equivalent to the base_map but for the temporary storage, this is only used when there is the necessity of
        restore a tensor (insert new data in the middle of a tensor).

    synchronizer: str
        Some Storages used to handle the files support a synchronizer, this parameter is used as a default
        synchronizer option for every one of them (you can pass different synchronizer to every tensor).
        The Mapping class provided by this library offers a lock solution for this purpose.

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
    the docs :meth:`TensorClient.read_from_formula`):

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
            base_map: MutableMapping,
            tmp_map: MutableMapping = None,
            synchronizer: str = None,
            **kwargs
    ):
        self.base_map = base_map
        if not isinstance(base_map, Mapping):
            self.base_map: Mapping = Mapping(base_map)

        self.tmp_map = self.base_map.sub_map('tmp') if tmp_map is None else tmp_map
        if not isinstance(self.tmp_map, Mapping):
            self.tmp_map: Mapping = Mapping(tmp_map)

        self.synchronizer = synchronizer
        self._tensors_definition = JsonStorage(
            base_map=self.base_map.sub_map('_tensors_definition'),
            tmp_map=self.tmp_map.sub_map('_tensors_definition'),
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
        self._tensors_definition.store(path=definition.path, new_data=definition.dict(exclude_unset=True))

    @validate_arguments
    def upsert_tensor(self, definition: TensorDefinition):
        """
        Upsert the definition of tensor, which is equivalent to the creation of the tensor but without data,
        in case that the tensor already exists it will be updated.

        Parameters
        ----------
        definition: TensorDefinition
            Read the docs of the `TensorDefinition` class for more info of the definition.

        """
        self._tensors_definition.upsert(path=definition.path, new_data=definition.dict())

    @validate_arguments
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        """
        Retrieve a tensor definition.

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
            raise KeyError(f'The tensor {path} has not been created using the create_tensor method')

    @validate_arguments
    def update_tensor_metadata(self, path: str, new_metadata: Dict[str, Any]):
        tensor_definition = self.get_tensor_definition(path)
        tensor_definition.metadata.update(new_metadata)
        self.upsert_tensor(tensor_definition)

    def get_all_tensors_definition(self) -> List[TensorDefinition]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self.get_tensor_definition,
                list(self._tensors_definition.base_map.keys())
            ))
        return results

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
    def delete_tensors(self, paths: List[str], only_data: bool = False):
        """
        Delete multiple tensors, in case that some of them does not exist it is not going to raise an error.

        Parameters
        ----------
        paths: List[str]
            paths of the tensors

        only_data: bool, default False
            If this option is marked as True only the data will be erased and not the definition

        """

        for path in paths:
            try:
                self.delete_tensor(path, only_data=only_data)
            except KeyError:
                pass

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
            base_map=self.base_map.sub_map(definition.path),
            tmp_map=self.tmp_map.sub_map(definition.path),
            **definition.storage.dict(exclude_unset=True)
        )
        return storage

    @staticmethod
    def _exec_on_dask(
            func: Callable,
            params,
            process: bool,
            *prev_tasks,
    ):
        if process:
            try:
                return func(**params)
            except Exception as e:
                e.args = (f"Tensor path: {params['path']}", *e.args)
                raise

    @staticmethod
    def exec_on_parallel(
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
            for sub_paths in mit.chunked(paths, max_parallelization):
                logger.info(f"Processing the following tensors: {sub_paths}")
                futures = [
                    pool.submit(
                        TensorClient._exec_on_dask,
                        method,
                        {"path": path, "compute": compute, **paths_kwargs[path]},
                        True
                    )
                    for path in sub_paths
                ]
                futures = [future.result() for future in futures]

                if not compute:
                    client.compute(futures, **compute_kwargs)

    def exec_on_dag_order(
            self,
            method: Union[Literal['append', 'update', 'store', 'upsert'], Callable],
            kwargs_groups: Dict[str, Dict[str, Any]] = None,
            tensors_path: List[str] = None,
            parallelization_kwargs: Dict[str, Any] = None,
            max_parallelization_per_group: Dict[str, int] = None,
            autofill_dependencies: bool = False,
            only_on_groups: set = None,
            check_dependencies: bool = True,
            omit_first_n_levels: int = 0,
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

        autofill_dependencies: bool, default False
            Automatically fill the dependencies of every tensor, useful to update some specific tensors and all their
            dependencies

        check_dependencies: bool, default True
            If True, will check if the dependencies are present in the DAG and raise a KeyError if not

        only_on_groups: set, default None
            Useful for filters the tensors base on the DAG groups (read the DAG models of tensordb)

        omit_first_n_levels: int, default 0
            Omit the first N levels of the DAG, useful in cases that all the previous steps were executed
            successfully, and the next one failed.

        max_parallelization_per_group: Dict[str, int] = None
            Sometimes there are groups that download all the data from remote sources with limited resources,
            so only one tensor or a limited number of them can be executed at the same time to avoid overloading
            the resources.

        """
        kwargs_groups = kwargs_groups or {}
        parallelization_kwargs = parallelization_kwargs or {}
        max_parallelization_per_group = max_parallelization_per_group or {}
        method = getattr(self, method) if isinstance(method, str) else method

        if tensors_path is None:
            tensors = [tensor for tensor in self.get_all_tensors_definition() if tensor.dag is not None]
        else:
            tensors = [self.get_tensor_definition(path) for path in tensors_path]
            if autofill_dependencies:
                tensors = dag.add_dependencies(tensors, self.get_all_tensors_definition())

        for i, level in enumerate(dag.get_tensor_dag(tensors, check_dependencies)):
            if i < omit_first_n_levels:
                continue

            logger.info(f'Executing the {i} level of the DAG')
            # Filter the tensors base on the omit parameter
            level = [tensor for tensor in level if method.__name__ not in tensor.dag.omit_on]
            if only_on_groups:
                # filter the invalid groups
                level = [tensor for tensor in level if tensor.dag.group in only_on_groups]
            if not level:
                continue

            for tensors in groupby_chunks(level, max_parallelization_per_group, lambda tensor: tensor.dag.group):
                self.exec_on_parallel(
                    method=method,
                    paths_kwargs={tensor.path: kwargs_groups.get(tensor.dag.group, {}) for tensor in tensors},
                    **parallelization_kwargs
                )

    def get_dag_for_dask(
            self,
            method: Union[Literal['append', 'update', 'store', 'upsert'], Callable],
            kwargs_groups: Dict[str, Dict[str, Any]] = None,
            tensors: List[TensorDefinition] = None,
            max_parallelization_per_group: Dict[str, int] = None,
            map_paths: Dict[str, str] = None,
            task_prefix: str = 'task-',
            final_task_name: str = 'WAIT',
    ):
        """
        This method was designed to create a Dask DAG for the given method, this is useful for parallelization
        of the execution of the tensors. The exec on dag order will be deprecated in the future.
        """
        kwargs_groups = kwargs_groups or {}
        map_paths = map_paths or {}
        max_parallelization_per_group = max_parallelization_per_group or {}
        method = getattr(self, method) if isinstance(method, str) else method

        if tensors is None:
            tensors = self.get_all_tensors_definition()

        tensors = [tensor for tensor in tensors if tensor.dag is not None]
        groups = {tensor.path: tensor.dag.group for tensor in tensors}
        # The new dependencies are applied to limit the amount of tasks processed in parallel
        # on each group
        new_dependencies = {}

        for level in dag.get_tensor_dag(tensors, False):
            prev_dependencies = set()
            for i, (name, group) in enumerate(iter_by_group_chunks(
                    level, max_parallelization_per_group, lambda tensor: tensor.dag.group
            )):
                if i != 0:
                    new_dependencies.update({tensor.path: prev_dependencies for tensor in group})
                prev_dependencies = set(tensor.path for tensor in group)

        graph = {}
        for tensor in tensors:
            path = tensor.path
            depends = set(tensor.dag.depends) | new_dependencies.get(path, set())
            group = groups[path]
            parameters = kwargs_groups.get(group, {}).copy()
            parameters['path'] = path
            graph[map_paths.get(path, task_prefix + path)] = (
                TensorClient._exec_on_dask,
                method,
                parameters,
                method.__name__ not in tensor.dag.omit_on,
                *tuple(map_paths.get(p, task_prefix + p) for p in depends)
            )

        graph[final_task_name] = (lambda *prev_tasks: None, *tuple(graph.keys()))
        return graph

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

    def read(
            self,
            path: Union[str, TensorDefinition, xr.DataArray, xr.Dataset],
            **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with read as method_name (has the same parameters).

        Returns
        -------
        An xr.DataArray that allow to read the data in the path.

        """
        if isinstance(path, (xr.DataArray, xr.Dataset)):
            return path

        return self.storage_method_caller(path=path, method_name='read', parameters=kwargs)

    def append(self, path: Union[str, TensorDefinition], **kwargs) -> List[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with append as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='append', parameters=kwargs)

    def update(self, path: Union[str, TensorDefinition], **kwargs) -> AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with update as method_name (has the same parameters).

        Returns
        -------
        Returns a single the AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='update', parameters=kwargs)

    def store(self, path: Union[str, TensorDefinition], **kwargs) -> AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with store as method_name (has the same parameters).

        Returns
        -------
        Returns a single the AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='store', parameters=kwargs)

    def upsert(self, path: Union[str, TensorDefinition], **kwargs) -> List[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with upsert as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='upsert', parameters=kwargs)

    def drop(self, path: Union[str, TensorDefinition], **kwargs) -> List[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with drop as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(path=path, method_name='drop', parameters=kwargs)

    def exist(self, path: str, only_definition: bool = False, **kwargs) -> bool:
        """
        Calls :meth:`TensorClient.storage_method_caller` with exist as method_name (has the same parameters).

        Returns
        -------
        A bool indicating if the file exist or not (True means yes).
        """
        try:
            exist_definition = self._tensors_definition.exist(path)
            if only_definition:
                return exist_definition
            return exist_definition and self.get_storage(path).exist(**kwargs)
        except KeyError:
            return False

    def get_cached_storage(
            self,
            path,
            max_cached_in_dim: int,
            dim: str,
            sort_dims: List[str],
            merge_cache: bool = False,
            update_logic: Literal["keep_last", "combine_first"] = "combine_first",
            **kwargs
    ):
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

        sort_dims: List[str]

        merge_cache: bool = False

        update_logic: Literal["keep_last", "combine_first"] = "combine_first"

        **kwargs
            Parameters used for the internal Storage that you chose.

        Returns
        -------
        A `CachedStorage` object.

        """
        return CachedStorage(
            storage=self.get_storage(path, **kwargs),
            max_cached_in_dim=max_cached_in_dim,
            dim=dim,
            sort_dims=sort_dims,
            merge_cache=merge_cache,
            update_logic=update_logic
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

        Another significant chracteristic is that you can even pass entiere python codes to create this new tensors
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
        for path in extract_paths_from_formula(formula):
            if original_path is not None and original_path == path:
                if storage is None:
                    raise ValueError(f'You can not make a self read without sending the storage parameter')
                data_fields[path] = storage.read()
            else:
                data_fields[path] = self.read(path)

        for path, dataset in data_fields.items():
            formula = formula.replace(f"`{path}`", f"data_fields['{path}']")

        formula_globals = {
            'xr': xr, 'np': np, 'pd': pd, 'da': da, 'dask': dask, 'self': self
        }
        kwargs.update({'data_fields': data_fields})

        if use_exec:
            exec(formula, formula_globals, kwargs)
            return kwargs['new_data']
        return eval(formula, formula_globals, kwargs)
