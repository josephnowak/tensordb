import itertools as it
import more_itertools as mit

from typing import Dict, Callable, Generator, Iterable


def groupby_chunks(
        iterable: Iterable,
        group_chunk_size: Dict,
        group_func: Callable,
        sort_func: Callable = None
) -> Generator:
    """
    This function apply a groupby over the iterable, and then it chunks the groups to iterate over them in order
    creating new iterables that looks as follows:
    [chunk0_group0, chunk0_group1, chunk0_group2] and [chunk1_group0, chunk1_group1, chunk1_group2]
    then those list of lists (list of chunks) are joined using itertools chain creating a unique list.

    This is useful for parallelize tasks with restrictions.

    Parameters
    ----------

    iterable:
        The iterable to group.

    group_chunk_size: Dict
        Size of the chunks of every group.

    group_func: Callable
        Function to group the iterable, equivalent of itertools.groupby key.

    sort_func: Callable, default None
        Function to sort the iterable (equivalent of sorted key), by default it is equal to the group_func.
        This is useful for preserve a specific order inside every group
    """
    sort_func = group_func if sort_func is None else sort_func
    return (
        # Filter the chunks with no data and then join all the chunks into a unique list.
        list(it.chain(*filter(None, tensors)))
        # Iterate in order over the chunked groups, this will generate a combinations like
        # [chunk0_group0, chunk0_group1, chunk0_group2] and [chunk1_group0, chunk1_group1, chunk1_group2].
        for tensors in it.zip_longest(*(
            # chunk the group based on the group_chunk_size size
            list(mit.chunked(group, group_chunk_size.get(name, None)))
            # group the data
            for name, group in it.groupby(
                sorted(iterable, key=sort_func),
                group_func
            )
        ))
    )
