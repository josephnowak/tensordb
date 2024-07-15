from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class DAGOrder(BaseModel):
    """
    As the tensor has relations between them, it is possible to create a DAG that facilities the process
    of updating every one of them, so the idea is that this class keep the necessary information for sorting the
    tensor execution.
    """

    depends: Optional[List[str]] = Field(
        default=[],
        title="Depends",
        description="Every element of the list is a tensor of which this tensor depends,"
                    "so this can be seen like every tensor Ai of the list has an edge to this tensor B, "
                    "so Ai->B for every i.",
    )
    group: str = Field(
        title="Group",
        default="regular",
        description="Useful to send parameters by groups on the exec_on_dag_order method of the tensor client",
    )
    omit_on: List[Literal["append", "update", "store", "upsert"]] = Field(
        title="Omit on",
        default=[],
        description="Indicate if the tensor must be omitted on the methods inside the list, "
                    "useful for generating reference nodes or avoid calling methods on tensors "
                    "that are simply formulas but other tensors use them.",
    )
