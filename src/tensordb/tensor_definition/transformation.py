from typing import Dict, List, Any, Optional, Literal, Annotated, Union

from pydantic import ConfigDict, BaseModel, Field
from tensordb.common import STORAGE_ACTIONS


class Method(BaseModel):
    """
    This class is used to indicate the method that want to be used on the sequential transformation
    """

    method_name: str = Field(
        title="Method Name",
        description="Name of the method that want to be used, it must match "
                    "with the methods of storage or the tensor_client",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        title="Parameters",
        default={},
        description="Default parameters for the execution of the method, they will be overwritten "
                    "if you sent other parameters during the call of the method",
    )
    result_name: Optional[str] = Field(
        None,
        title="Result Name",
        description="Indicate the name of the output that the method produce.",
    )


class SequentialTransformation(BaseModel):
    transformation_type: Literal["sequential"] = "sequential"
    methods: List[Method] = Field(
        title="Methods",
        description="Every element of the list must contain a Method "
                    "object which specify the method that must be executed.",
    )
    # model_config = ConfigDict(extra="allow")


class SubstituteTransformation(BaseModel):
    transformation_type: Literal["substitute"] = "substitute"
    substitute_method: STORAGE_ACTIONS = Field(
        title="Substitute Method",
        description="Replace the original method by this one, for example "
                    "you can modify that every time you call append, "
                    "it will call the append method.",
    )
    # model_config = ConfigDict(extra="allow")


Transformation = Annotated[
    Union[SequentialTransformation, SubstituteTransformation],
    Field(discriminator="transformation_type")
]
