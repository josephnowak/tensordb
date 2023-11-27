import itertools as it
from typing import Callable, Union, Dict, Any, List, Hashable
from typing import Generator, Iterable

import more_itertools as mit
import numpy as np
import xarray as xr
from dask import array as da
from pydantic import validate_call


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
    )))
    )


def iter_by_group_chunks(
        iterable: Iterable,
        group_chunk_size: Dict,
        group_func: Callable,
) -> Generator:
    for name, group in it.groupby(
            sorted(iterable, key=group_func),
            group_func
    ):
        for chunk in mit.chunked(group, group_chunk_size.get(name, None)):
            yield name, chunk


def extract_paths_from_formula(formula) -> set:
    paths_intervals = np.array([i for i, c in enumerate(formula) if c == '`'])
    paths = {
        formula[paths_intervals[i] + 1: paths_intervals[i + 1]]
        for i in range(0, len(paths_intervals), 2)
    }
    return paths


def empty_xarray(dims, coords, chunks, dtype):
    return xr.DataArray(
        da.empty(
            shape=tuple(len(coords[dim]) for dim in dims),
            dtype=dtype,
            chunks=chunks
        ),
        dims=dims,
        coords=coords
    )


@validate_call(config=dict(arbitrary_types_allowed=True))
def xarray_from_func(
        func: Callable,
        dims: List[Hashable],
        coords: Dict[Hashable, Union[List, np.ndarray]],
        chunks: Union[List[Union[int, None]], Dict[Hashable, int]],
        dtypes: Union[List[Any], Any],
        data_names: Union[List[Hashable], str] = None,
        func_parameters: Dict[str, Any] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Equivalent of dask fromfunction but it sends the xarray coords of every chunk instead of the positions

    The most useful part of this function is that you can pivot a table faster (you already has the coords, why pivot
    as if you don't know the shape of your tensor?). The idea is that your function make the pivot of the data,
    but the function in fact will be only doing a pivot of a chunk with a known shape (a small portion of the data)
    and with dask the chunks are processed in parallel, which is ideal for cases when you have to query a slow DB.

    Parameters
    ----------

    func: Callable
        The function must return an array with the exact shape of the coords (same order too), in case that the
        data_names is active, the function must return a list where every element of the list must be also an array.

    dims: List[Hashable]
        Dimensions of your dataset or data array (read the docs of Xarray DataArray or Dataset for more info)

    coords: Dict[Hashable, Union[List, np.ndarray]]
        Coords of your dataset or data array (read the docs of Xarray DataArray or Dataset for more info).

        For relational databases is useful to use a query with a Distinct over the columns to get the coords.

    chunks: Union[List[Union[int, None]], Dict[Hashable, int]]
        The chunks indicate how to divide the array into multiple parts (read the docs of Dask for more info)
        Internally it create chunks based on the coords.

    dtypes: Union[List[Any], Any]
        Indicate the dtype for every DataArray inside your Dataset, in case of sent a unique dtype
        the array will be considered as a DataArray instead of a Dataset.
        For multiple dtypes the results of the func must be aligned with the dtypes in other
        case it will raise a Dask error.

    data_names: List[Hashable], default None
        Indicate the names of the different DataArray inside your Dataset.
        The data_names must be aligned with dtypes, in other case it will raise an Error.

    func_parameters: Dict[str, Any], default None
        Extra parameters for the function

    """
    if isinstance(chunks, dict):
        chunks = [chunks[dim] for dim in dims]
    chunks = [len(coords[dim]) if chunk is None else chunk for chunk, dim in zip(chunks, dims)]
    func_parameters = {} if func_parameters is None else func_parameters

    if data_names is None or isinstance(data_names, str):
        arr = empty_xarray(dims, coords, chunks, dtypes)
    else:
        if len(dtypes) != len(data_names):
            raise ValueError(
                f'The number of dtypes ({len(dtypes)}) does not match the number of dataset names '
                f'({len(data_names)}), you need to specify a dtype for every data array in your dataset'
            )
        arr = xr.Dataset({
            name: empty_xarray(dims, coords, chunks, dtype)
            for name, dtype in zip(data_names, dtypes)
        }, coords=coords)

    return arr.map_blocks(
        func,
        kwargs=func_parameters,
        template=arr
    )
