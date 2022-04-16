from functools import reduce

from typing import Dict, List, Set, Any, Generator
from loguru import logger

from tensordb.tensor_definition import TensorDefinition


# TODO: Update to python 3.9 or 3.10 to start using graphlib instead of use this function


def get_tensor_dag(
        tensors: List[TensorDefinition],
        check_dependencies: bool
) -> List[List[TensorDefinition]]:
    tensor_search = {tensor.path: tensor for tensor in tensors}
    # Create the dag based on the dependencies, so the node used as Key depends on the Nodes in the values
    # It's like there is an array from every node in the values to the Key node
    dag = {tensor.path: set(tensor.dag.depends) for tensor in tensors}

    ordered_tensors = []
    extra_items_in_deps = reduce(set.union, dag.values()) - set(dag.keys())
    dag.update({item: set() for item in extra_items_in_deps})
    # Group the tensors based in the execution order created by the dag
    while True:
        ordered = set(item for item, dependencies in dag.items() if not dependencies)
        if not ordered:
            break

        ordered_tensors.append([
            tensor_search[path]
            for path in ordered
            if check_dependencies or path in tensor_search
        ])

        dag = {
            item: dependencies - ordered for item, dependencies in dag.items()
            if item not in ordered
        }
    if dag:
        raise ValueError(
            f'There is a cyclic dependency between the tensors, '
            f'the key is the node and the values are the dependencies: {dag}'
        )

    return ordered_tensors


def add_dependencies(
        tensors: List[TensorDefinition],
        total_tensors: List[TensorDefinition]
) -> List[TensorDefinition]:
    total_tensors_search = {tensor.path: tensor for tensor in total_tensors}
    total_paths = set(tensor.path for tensor in tensors)
    for path, tensor in total_tensors_search.items():
        if path not in total_paths:
            continue
        total_paths.update(tensor.dag.depends)
    return [total_tensors_search[path] for path in total_paths]
