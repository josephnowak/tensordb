from pydantic import BaseModel, Extra, Field

from typing import Dict, List, Any, Optional, Literal


class DAGOrder(BaseModel):
    """
    As the tensor has relations between them it is possible to create a DAG that facilities the process
    of updating everyone of them, so the idea is that this class keep the necessary information for sorting the
    tensor execution.

    Parameters
    ----------

    depends: Optional[List[str]]
        Every element of the list is a tensor on which this tensor depends, so this can be seen like every tensor Ai
        of the list has an edge to this tensor B, so Ai->B for every i.
    """
    depends: Optional[List[str]]


class StorageDefinition(BaseModel):
    """
    Definition of the storage of the tensor

    Parameters
    ----------

    storage_name, default 'zarr_storage': Optional[Literal['json_storage', 'zarr_storage']]
        Indicate which data storage want to be used.

    synchronizer, default None: Optional[Literal['process', 'thread']]
        Type of synchronizer use in the storage (read the docs of the storages).
    """
    storage_name: Optional[Literal['json_storage', 'zarr_storage']] = 'zarr_storage'
    synchronizer: Optional[Literal['process', 'thread']] = None

    class Config:
        extra = Extra.allow


class MethodDescriptor(BaseModel):
    """
    This class is used to indicate the method that want to be transformed during the data_transformation process

    Parameters
    ----------

    method_name: str
        Name of the method that want to be use, it must match with the methods of storage or the tensor_client

    parameters: Dict[str, Any]
        Default parameters for the execution of the method, they will be overwrited if you sent other parameters
        during the call of the method

    result_name, default 'new_data': Optional[str]
        Indicate the name of the output that the method produce.

    """
    method_name: str
    parameters: Optional[Dict[str, Any]] = {}
    result_name: Optional[str]


class Definition(BaseModel):
    """
    The definition allows to modify/extend the default behaviour of every method on the tensor_client, this is useful
    because you can create a pipeline that transform or read your data from other sources.

    Parameters
    ----------

    data_transformation: Optional[List[MethodDescriptor]]
        Every element of the list must contain a MethodDescriptor object which specify which method must be executed.

    substitute_method: Optional[str]
        Replace the original method by this one, for example you can modify that every time you call append,
        it will call the append method.

    """

    data_transformation: Optional[List[MethodDescriptor]]
    substitute_method: Optional[str]

    class Config:
        extra = Extra.allow


class TensorDefinition(BaseModel):
    """
    The tensor definition is the core of tensordb, it allows to add functionalities to every tensor that are
    kept over the time, so you don't have to know before hand how to transform every tensor.

    Parameters
    ----------

    path: str
        Path where the tensor is going to be stored (it's concatenated to the base_map of tensor_client)

    definition: Dict[str, Definition]
        The key indicate to which method must be applied the definition (read the doc of Definition)

    dag: Optional[DAGOrder]
        Indicate the relations/dependencies that a tensor has with others, useful for executing an operation over
        multiples tensor that has dependencies

    storage: Optional[StorageDefinition]
        Useful to send parameters to the Storage constructor or to change the default data storage

    metadata: Optional[Dict]
        Metadata of the tensor
    """

    path: str
    definition: Dict[str, Definition]
    dag: Optional[DAGOrder]
    storage: Optional[StorageDefinition] = StorageDefinition()
    metadata: Optional[Dict]

    def __hash__(self):
        return hash(self.path)


