from concurrent.futures import ThreadPoolExecutor
from typing import Any

import obstore
import orjson
import xarray as xr
from pydantic import validate_call
from xarray.backends.common import AbstractWritableDataStore

from tensordb.algorithms import Algorithms
from tensordb.clients.base import BaseTensorClient
from tensordb.storages import MAPPING_STORAGES, BaseStorage, JsonStorage
from tensordb.tensor_definition import TensorDefinition
from tensordb.utils.ic_storage_model import LocalStorageModel, S3StorageModel


class TensorClient(BaseTensorClient, Algorithms):
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
    base_map: FsspecStore
       Mapping storage interface where all the tensors are stored.

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

    internal_actions = ["store", "update", "append", "upsert", "drop"]

    def __init__(
        self,
        ob_store: obstore.store.ObjectStore,
        ic_storage: S3StorageModel | LocalStorageModel = None,
        **kwargs,
    ):
        self.ob_store = ob_store

        self._tensors_definition = JsonStorage(
            ob_store=ob_store,
            sub_path="_tensors_definition",
        )
        self.ic_storage = ic_storage

    def add_custom_data(self, path, new_data: dict):
        self.ob_store.put(
            path, orjson.dumps(new_data, option=orjson.OPT_SERIALIZE_NUMPY)
        )

    def get_custom_data(self, path, default=None):
        try:
            return orjson.loads(self.ob_store.get(path).bytes().to_bytes())
        except FileNotFoundError:
            return default

    @validate_call
    def create_tensor(self, definition: TensorDefinition):
        """
        Store the definition of tensor, which is equivalent to the creation of the tensor but without data

        Parameters
        ----------
        definition: TensorDefinition
            Read the docs of the `TensorDefinition` class for more info of the definition.

        """
        self._tensors_definition.store(
            path=definition.path, new_data=definition.model_dump(exclude_unset=True)
        )

    @validate_call
    def upsert_tensor(self, definition: TensorDefinition):
        """
        Upsert the definition of tensor, which is equivalent to the creation of the tensor but without data,
        in case that the tensor already exists it will be updated.

        Parameters
        ----------
        definition: TensorDefinition
            Read the docs of the `TensorDefinition` class for more info of the definition.

        """
        self._tensors_definition.upsert(
            path=definition.path, new_data=definition.model_dump()
        )

    @validate_call
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
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The tensor {path} has not been created using the create_tensor method"
            ) from e

    @validate_call
    def update_tensor_metadata(self, path: str, new_metadata: dict[str, Any]):
        tensor_definition = self.get_tensor_definition(path)
        tensor_definition.metadata.update(new_metadata)
        self.upsert_tensor(tensor_definition)

    def get_all_tensors_definition(self) -> list[TensorDefinition]:
        with ThreadPoolExecutor() as executor:
            paths = self.ob_store.list(
                prefix=self._tensors_definition.sub_path,
            ).collect()
            paths = [
                path["path"].replace("\\", "/").replace("_tensors_definition/", "")
                for path in paths
            ]
            results = list(
                executor.map(
                    self.get_tensor_definition,
                    paths,
                )
            )
        return results

    @validate_call
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

    @validate_call
    def get_storage(self, path: str | TensorDefinition) -> BaseStorage:
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

        storage = MAPPING_STORAGES[definition.storage.storage_name]
        storage = storage(
            ob_store=self.ob_store,
            sub_path=definition.path,
            ic_storage=self.ic_storage,
            **definition.storage.model_dump(exclude_unset=True),
        )
        return storage

    def read(
        self, path: str | TensorDefinition | xr.DataArray | xr.Dataset, **kwargs
    ) -> xr.DataArray | xr.Dataset:
        """
        Calls :meth:`TensorClient.storage_method_caller` with read as method_name (has the same parameters).

        Returns
        -------
        An xr.DataArray that allow to read the data in the path.

        """
        if isinstance(path, (xr.DataArray, xr.Dataset)):
            return path

        return self.storage_method_caller(
            path=path, method_name="read", parameters=kwargs
        )

    def append(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with append as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(
            path=path, method_name="append", parameters=kwargs
        )

    def update(
        self, path: str | TensorDefinition, **kwargs
    ) -> AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with update as method_name (has the same parameters).

        Returns
        -------
        Returns a single the AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(
            path=path, method_name="update", parameters=kwargs
        )

    def store(
        self, path: str | TensorDefinition, **kwargs
    ) -> AbstractWritableDataStore:
        """
        Calls :meth:`TensorClient.storage_method_caller` with store as method_name (has the same parameters).

        Returns
        -------
        Returns a single the AbstractWritableDataStore object,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(
            path=path, method_name="store", parameters=kwargs
        )

    def upsert(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with upsert as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(
            path=path, method_name="upsert", parameters=kwargs
        )

    def drop(
        self, path: str | TensorDefinition, **kwargs
    ) -> list[AbstractWritableDataStore]:
        """
        Calls :meth:`TensorClient.storage_method_caller` with drop as method_name (has the same parameters).

        Returns
        -------
        Returns a List of AbstractWritableDataStore objects,
        which is used as an interface for the corresponding backend that you select in xarray (the Storage).

        """
        return self.storage_method_caller(
            path=path, method_name="drop", parameters=kwargs
        )

    def exist(self, path: str, only_definition: bool = False, **kwargs) -> bool:
        """
        Calls :meth:`TensorClient.storage_method_caller` with exist as method_name (has the same parameters).

        Returns
        -------
        A bool indicating if the file exist or not (True means yes).
        """
        try:
            exist_definition = self._tensors_definition.exist(path)
            if only_definition or (not exist_definition):
                return exist_definition
            self.read(path=path, **kwargs)
            return True
        except FileNotFoundError:
            return False

    def read_from_formula(
        self,
        formula: str,
        use_exec: bool = False,
        original_path: str = None,
        storage: BaseStorage = None,
        **kwargs: dict[str, Any],
    ) -> xr.DataArray | xr.Dataset:
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

        return super().read_from_formula(
            formula=formula,
            use_exec=use_exec,
            original_path=original_path,
            storage=storage,
            **kwargs,
        )
