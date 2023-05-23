from functools import reduce
from typing import List

import more_itertools as mit
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


def get_leaf_tasks(tensors, new_dependencies=None):
    # Add the non blocking tasks to a final task
    final_tasks = set(tensor.path for tensor in tensors)
    final_tasks -= set().union(*[
        set(tensor.dag.depends) | new_dependencies.get(tensor.path, set())
        for tensor in tensors
    ])
    return final_tasks


def get_limit_dependencies(total_tensors, max_parallelization_per_group):
    new_dependencies = {}
    group_tensors = {}
    for v in total_tensors:
        if v.dag.group not in group_tensors:
            group_tensors[v.dag.group] = []
        group_tensors[v.dag.group].append(v)

    for group, limit in max_parallelization_per_group.items():
        if group not in group_tensors:
            continue
        tensors = group_tensors[group]
        for level in get_tensor_dag(tensors, False):
            if not level:
                continue
            level = sorted([tensor.path for tensor in level])
            level = list(mit.sliced(level, limit))
            prev_dependencies = set(level[0])
            for act_tensors in level[1:]:
                new_dependencies.update({tensor: prev_dependencies for tensor in act_tensors})
                prev_dependencies = set(act_tensors)
    return new_dependencies
