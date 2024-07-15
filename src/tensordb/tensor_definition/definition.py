from typing import Dict, List, Any, Optional, Literal, Annotated, Union

from pydantic import ConfigDict, BaseModel, Field

from tensordb.tensor_definition.transformation import Transformation
from tensordb.tensor_definition.dag import DAGOrder
from tensordb.backends.engines import ENGINES
from tensordb.common import STORAGE_ACTIONS


class TensorDefinition(BaseModel):
    """
    The tensor definition is the core of tensordb, it allows to standardize the way that the database read and write
    the files that share a specific structure.
    """

    path: str = Field(
        title="Path",
        description="Path where the tensor is going to be stored "
                    "(it's concatenated to the base_map of tensor_client)",
    )
    transformation: Optional[
        Dict[STORAGE_ACTIONS, Transformation]
    ] = Field(
        title="Transformation",
        default={},
        description="The key indicate to which method must be applied the transformation "
                    "(read the doc of Definition)",
    )
    dag: Optional[DAGOrder] = Field(
        default=DAGOrder(),
        title="DAG",
        description="Indicate the relations/dependencies that a tensor has with others, "
                    "useful for executing an operation over multiple tensors that has dependencies",
    )
    # backend: Union[tuple(ENGINES.values())] = Field(
    #     title="Backend",
    #     description="Indicates the backend used to read and write the data on the path specified",
    #     discriminator="backend_name",
    #     default=ENGINES["zarr"](data_name="data")
    # )
    metadata: Optional[Dict] = Field(
        title="Metadata",
        default={},
        description="Metadata of the tensor",
    )

    def __hash__(self):
        return hash(self.path)
