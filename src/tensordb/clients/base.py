import abc
from collections.abc import Callable
from typing import Any, Literal

import dask
import dask.array as da
import more_itertools as mit
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from dask.highlevelgraph import HighLevelGraph
from loguru import logger
from pydantic import validate_call
from xarray.backends.common import AbstractWritableDataStore

from tensordb.algorithms import Algorithms
from tensordb.storages import BaseStorage, CachedStorage
from tensordb.tensor_definition import Definition, MethodDescriptor, TensorDefinition
from tensordb.utils import dag
from tensordb.utils.method_inspector import get_parameters
from tensordb.utils.tools import extract_paths_from_formula, groupby_chunks


class BaseTensorClient(Algorithms, abc.ABC):
    internal_actions = ["store", "update", "append", "upsert", "drop"]

    def add_custom_data(self, path, new_data: dict):
        pass

    def get_custom_data(self, path, default=None):
        pass

    @abc.abstractmethod
    def create_tensor(self, definition: TensorDefinition):
        pass

    @abc.abstractmethod
    def upsert_tensor(self, definition: TensorDefinition):
        pass

    @abc.abstractmethod
    def get_tensor_definition(self, path: str) -> TensorDefinition:
        pass

    @validate_call
    def update_tensor_metadata(self, path: str, new_metadata: dict[str, Any]):
        tensor_definition = self.get_tensor_definition(path)
        tensor_definition.metadata.update(new_metadata)
        self.upsert_tensor(tensor_definition)

    @abc.abstractmethod
    def get_all_tensors_definition(self) -> list[TensorDefinition]:
        pass

    @abc.abstractmethod
    def delete_tensor(self, path: str, only_data: bool = False) -> Any:
        pass

    @validate_call
    def delete_tensors(self, paths: list[str], only_data: bool = False):
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
            except FileNotFoundError:
                pass

    @abc.abstractmethod
    def get_storage(self, path: str | TensorDefinition) -> BaseStorage:
        pass

    @staticmethod
    def _exec_on_dask(
        func: Callable,
        params,
        *prev_tasks,
    ):
        try:
            return func(**params)
        except Exception as e:
            e.args = (f"Tensor path: {params['path']}", *e.args)
            raise e

    @staticmethod
    def exec_on_parallel(
        method: Callable,
        paths_kwargs: dict[str, dict[str, Any]],
        max_parallelization: int = None,
        client: Client = None,
        compute_kwargs: dict[str, Any] = None,
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

        client: Client, default None
            Dask client, useful for in combination with the compute parameter

        compute_kwargs: Dict[str, Any], default None
            Parameters of the dask.compute or client.compute
        """
        paths = list(paths_kwargs.keys())

        max_parallelization = (
            np.inf if max_parallelization is None else max_parallelization
        )
        max_parallelization = min(max_parallelization, len(paths))
        compute_kwargs = compute_kwargs or {}
        client = dask if client is None else client

        for sub_paths in mit.chunked(paths, max_parallelization):
            logger.info(f"Processing the following tensors: {sub_paths}")
            client.compute(
                [
                    dask.delayed(BaseTensorClient._exec_on_dask)(
                        func=method, params={"path": path, **paths_kwargs[path]}
                    )
                    for path in sub_paths
                ],
                **compute_kwargs,
            )

    def exec_on_dag_order(
        self,
        method: str | Callable,
        kwargs_groups: dict[str, dict[str, Any]] = None,
        tensors_path: list[str] = None,
        parallelization_kwargs: dict[str, Any] = None,
        max_parallelization_per_group: dict[str, int] = None,
        autofill_dependencies: bool = False,
        only_on_groups: set = None,
        check_dependencies: bool = True,
        omit_first_n_levels: int = 0,
    ):
        """
        This method was designed to execute multiple methods of the client on parallel and following the dag
        order, this is useful for update/store multiple tensors that have dependencies between them in a fast way.
        Internally calls the :meth:`BaseTensorClient.exec_on_parallel` for every level of the created DAG

        Parameters
        ----------

        method: Union[str, Callable]
            method of the tensor client that is going to be executed on parallel

        kwargs_groups: Dict[str, Dict[str, Any]]
            Kwargs sent to the method base on the DAG groups, read the docs of :meth:`DAGOrder` for more info

        tensors_path: List[str], default None
            Indicates the tensors on which the operation will be applied

        parallelization_kwargs: Dict[str, Any]
            Kwargs sent to :meth:`BaseTensorClient.exec_on_parallel`

        autofill_dependencies: bool, default False
            Automatically fill the dependencies with every tensor, useful to
            update some specific tensors and all their dependencies

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
            tensors = [
                tensor
                for tensor in self.get_all_tensors_definition()
                if tensor.dag is not None
            ]
        else:
            tensors = [self.get_tensor_definition(path) for path in tensors_path]
            if autofill_dependencies:
                tensors = dag.add_dependencies(
                    tensors, self.get_all_tensors_definition()
                )

        for i, level in enumerate(dag.get_tensor_dag(tensors, check_dependencies)):
            if i < omit_first_n_levels:
                continue

            logger.info(f"Executing the {i} level of the DAG")
            # Filter the tensors base on the omit parameter
            level = [
                tensor for tensor in level if method.__name__ not in tensor.dag.omit_on
            ]
            if only_on_groups:
                # filter the invalid groups
                level = [
                    tensor for tensor in level if tensor.dag.group in only_on_groups
                ]

            if not level:
                continue

            for tensors in groupby_chunks(
                level, max_parallelization_per_group, lambda tensor: tensor.dag.group
            ):
                self.exec_on_parallel(
                    method=method,
                    paths_kwargs={
                        tensor.path: kwargs_groups.get(tensor.dag.group, {})
                        for tensor in tensors
                    },
                    **parallelization_kwargs,
                )

    def get_dag_for_dask(
        self,
        method: str | Callable,
        kwargs_groups: dict[str, dict[str, Any]] = None,
        tensors: list[TensorDefinition] = None,
        max_parallelization_per_group: dict[str, int] = None,
        map_paths: dict[str, str] = None,
        task_prefix: str = "task-",
        final_task_name: str = "WAIT",
    ) -> HighLevelGraph:
        """
        This method was designed to create a Dask DAG for the given method, this is useful for parallelization
        of the execution of the tensors. The exec on dag order will be deprecated in the future.
        """
        kwargs_groups = kwargs_groups or {}
        map_paths = map_paths or {}
        max_parallelization_per_group = max_parallelization_per_group or {}
        method = getattr(self, method) if isinstance(method, str) else method
        none_func = lambda *x: None

        if tensors is None:
            tensors = self.get_all_tensors_definition()

        tensors = [tensor for tensor in tensors if tensor.dag is not None]
        groups = {tensor.path: tensor.dag.group for tensor in tensors}
        # The new dependencies are applied to limit the amount of tasks processed in parallel
        # on each group
        new_dependencies = dag.get_limit_dependencies(
            tensors, max_parallelization_per_group
        )

        graph = {}
        for tensor in tensors:
            path = tensor.path
            depends = set(tensor.dag.depends) | new_dependencies.get(path, set())
            params = kwargs_groups.get(groups[path], {})
            params["path"] = path
            func = (
                none_func
                if method.__name__ in tensor.dag.omit_on
                else self._exec_on_dask
            )
            graph[map_paths.get(path, task_prefix + path)] = (
                func,
                method,
                params,
                *tuple(map_paths.get(p, task_prefix + p) for p in depends),
            )

        final_tasks = dag.get_leaf_tasks(tensors, new_dependencies)
        final_tasks = tuple(
            map_paths.get(path, task_prefix + path) for path in final_tasks
        )

        graph[final_task_name] = (none_func, *tuple(final_tasks))
        return HighLevelGraph.from_collections(final_task_name, graph)

    def apply_data_transformation(
        self,
        data_transformation: list[MethodDescriptor],
        storage: BaseStorage,
        definition: dict[str, Definition],
        parameters: dict[str, Any],
        debug: bool = False,
    ):
        parameters = {**{"new_data": None}, **parameters}
        for descriptor in data_transformation:
            func = getattr(storage, descriptor.method_name, None)
            if func is None:
                func = getattr(self, descriptor.method_name)

            method_parameters = descriptor.parameters
            if descriptor.method_name in definition:
                method_parameters = {
                    **definition[descriptor.method_name].model_dump(exclude_unset=True),
                    **method_parameters,
                }
            result = func(**get_parameters(func, parameters, method_parameters))

            if descriptor.result_name is not None:
                parameters.update({descriptor.result_name: result})
            else:
                parameters.update(
                    result if isinstance(result, dict) else {"new_data": result}
                )

        return parameters["new_data"]

    @validate_call
    def storage_method_caller(
        self,
        path: str | TensorDefinition,
        method_name: str,
        parameters: dict[str, Any],
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
        tensor_definition = path
        if isinstance(path, str):
            tensor_definition = self.get_tensor_definition(path)
        definition = tensor_definition.definition

        storage = self.get_storage(path=tensor_definition)
        parameters.update(
            {
                "definition": definition,
                "original_path": path if isinstance(path, str) else path.path,
                "storage": storage,
            }
        )

        if method_name in definition:
            method_settings = definition[method_name]

            if method_settings.substitute_method is not None:
                func = getattr(self, method_settings.substitute_method)
                if func.__name__ in self.internal_actions:
                    return func(path=path, **parameters)
                return func(
                    **get_parameters(
                        func,
                        parameters,
                        definition[func.__name__].model_dump(exclude_unset=True)
                        if func.__name__ in definition
                        else {},
                    )
                )

            if method_settings.data_transformation is not None:
                parameters["new_data"] = self.apply_data_transformation(
                    data_transformation=method_settings.data_transformation,
                    storage=storage,
                    definition=definition,
                    parameters=parameters,
                )
                if method_name == "read":
                    return parameters["new_data"]

        func = getattr(storage, method_name)
        return func(**get_parameters(func, parameters))

    @abc.abstractmethod
    def read(
        self, path: str | TensorDefinition | xr.DataArray | xr.Dataset, **kwargs
    ) -> xr.DataArray | xr.Dataset:
        raise ValueError

    @abc.abstractmethod
    def append(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        pass

    @abc.abstractmethod
    def update(
        self, path: str | TensorDefinition, **kwargs
    ) -> AbstractWritableDataStore:
        pass

    @abc.abstractmethod
    def store(
        self, path: str | TensorDefinition, **kwargs
    ) -> AbstractWritableDataStore:
        pass

    @abc.abstractmethod
    def upsert(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        pass

    @abc.abstractmethod
    def drop(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        pass

    @abc.abstractmethod
    def exist(self, path: str, only_definition: bool = False, **kwargs) -> bool:
        pass

    def get_cached_storage(
        self,
        path,
        max_cached_in_dim: int,
        dim: str,
        sort_dims: list[str],
        merge_cache: bool = False,
        update_logic: Literal["keep_last", "combine_first"] = "combine_first",
        **kwargs,
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
            update_logic=update_logic,
        )

    def read_from_formula(
        self,
        formula: str,
        use_exec: bool = False,
        original_path: str = None,
        storage: BaseStorage = None,
        **kwargs: dict[str, Any],
    ) -> xr.DataArray | xr.Dataset:
        data_fields = {}
        for path in extract_paths_from_formula(formula):
            if original_path is not None and original_path == path:
                if storage is None:
                    raise ValueError(
                        "You can not make a self read without sending the storage parameter"
                    )
                data_fields[path] = storage.read()
            else:
                data_fields[path] = self.read(path)

        for path, _ in data_fields.items():
            formula = formula.replace(f"`{path}`", f"data_fields['{path}']")

        formula_globals = {
            "xr": xr,
            "np": np,
            "pd": pd,
            "da": da,
            "dask": dask,
            "self": self,
        }
        kwargs.update({"data_fields": data_fields})

        if use_exec:
            exec(formula, formula_globals, kwargs)
            return kwargs["new_data"]

        formula = formula.replace("\n", "")
        return eval(formula, formula_globals, kwargs)
