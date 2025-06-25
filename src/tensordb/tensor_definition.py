from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DAGOrder(BaseModel):
    """
    As the tensor has relations between them, it is possible to create a DAG that facilities the process
    of updating every one of them, so the idea is that this class keep the necessary information for sorting the
    tensor execution.
    """

    depends: list[str] | None = Field(
        default=[],
        title="Depends",
        description="""
        Every element of the list is a tensor on which this tensor depends, so this can be seen like every tensor Ai
        of the list has an edge to this tensor B, so Ai->B for every i.
        """,
    )
    group: str = Field(
        title="Group",
        default="regular",
        description="""
        Useful to send parameters by groups on the exec_on_dag_order method of the tensor client
        """,
    )
    omit_on: list[Literal["append", "update", "store", "upsert"]] = Field(
        title="Omit on",
        default=[],
        description="""
        Indicate if the tensor must be omitted on the methods inside the list,
        useful for generating reference nodes or avoid calling methods on tensors that are simply formulas
        but other tensors use them.
        """,
    )


class StorageDefinition(BaseModel):
    """
    Definition of the storage of the tensor
    """

    storage_name: Literal["json_storage", "zarr_storage"] | None = Field(
        title="Storage Name",
        default="zarr_storage",
        description="""
        Indicate which data storage want to be used.
        """,
    )
    model_config = ConfigDict(extra="allow")


class MethodDescriptor(BaseModel):
    """
    This class is used to indicate the method that want to be transformed during the data_transformation process
    """

    method_name: str = Field(
        title="Method Name",
        description="""
        Name of the method that want to be used, it must match with the methods of storage or the tensor_client
        """,
    )
    parameters: dict[str, Any] | None = Field(
        title="Parameters",
        default={},
        description="""
        Default parameters for the execution of the method, they will be overwritten if you sent other parameters
        during the call of the method
        """,
    )
    result_name: str | None = Field(
        None,
        title="Result Name",
        description="""
        Indicate the name of the output that the method produce.
        """,
    )


class Definition(BaseModel):
    """
    The definition allows to modify/extend the default behaviour of every method on the tensor_client, this is useful
    because you can create a pipeline that transform or read your data from other sources.
    """

    data_transformation: list[MethodDescriptor] | None = Field(
        title="Data Transformation",
        default=None,
        description="""
        Every element of the list must contain a MethodDescriptor object which specify the method that must be executed.
        """,
    )
    substitute_method: str | None = Field(
        title="Substitute Method",
        default=None,
        description="""
        Replace the original method by this one, for example you can modify that every time you call append,
        it will call the append method.
        """,
    )
    model_config = ConfigDict(extra="allow")


class TensorDefinition(BaseModel):
    """
    The tensor definition is the core of tensordb, it allows adding functionalities to every tensor that are
    kept over the time, so you don't have to know beforehand how to transform every tensor.
    """

    path: str = Field(
        title="Path",
        description="""
        Path where the tensor is going to be stored (it's concatenated to the base_map of tensor_client)
        """,
    )

    definition: dict[str, Definition] | None = Field(
        title="Definition",
        default={},
        description="""
        The key indicate to which method must be applied the definition (read the doc of Definition)
        """,
    )
    dag: DAGOrder | None = Field(
        default=DAGOrder(),
        title="DAG",
        description="""
        Indicate the relations/dependencies that a tensor has with others, useful for executing an operation over
        multiple tensors that has dependencies
        """,
    )
    storage: StorageDefinition | None = Field(
        title="Storage",
        default=StorageDefinition(),
        description="""
        Useful to send parameters to the Storage constructor or to change the default data storage
        """,
    )
    metadata: dict | None = Field(
        title="Metadata",
        default={},
        description="""
        Metadata of the tensor
        """,
    )

    def __hash__(self):
        return hash(self.path)
