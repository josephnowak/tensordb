import itertools
from math import ceil
from typing import Iterable, Callable, Union, Dict, Any, List, Hashable

import dask
import numpy as np
import xarray as xr
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def defined_translation(
        func: Callable,
        dims: List[Hashable],
        coords: Dict[Hashable, Union[List, np.ndarray]],
        chunks: List[Union[int, None]],
        dtypes: Union[List[Any], Any],
        data_names: List[Hashable] = None,
        func_parameters: Dict[str, Any] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Translate any kind of format that allow slice or indexing (ideal for normalized formats like the one used by
    relational databases) to Xarray.

    The idea behind this function is that it creates a set of chunks (not overlapping) of the coords that are use
    to call your func with dask delayed to then create a DataArray or Dataset for every chunk,
    after that all the chunks are combined into a unique Dataset or DataArray.

    The most useful part of this function is that you can pivot a table faster (you already has the coords, why pivot
    as if you don't know the shape of your tensor?). The idea is that your function make the pivot of the data,
    but the function in fact will be only doing a pivot of a chunk with a known shape (an small portion of the data)
    and with dask the chunks are processed in parallel, which is ideal for cases when you have to query an slow DB.

    This function will send the next parameters to your function:
        1. coords

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

    chunks: List[int]
        The chunks indicate how to divide the array into multiple parts (read the docs of Dask for more info)
        Internally it create chunks based on the coords.

    dtypes: Union[List[Any], Any]
        Indicate the dtype for every DataArray inside your Dataset, in case of sent a unique dtype
        the array will be consider as a DataArray instead of a Dataset.
        For multiple dtypes the results of the func must be aligned with the dtypes in other
        case it will raise a Dask error.

    data_names: List[Hashable], default None
        Indicate the names of the different DataArray inside your Dataset.
        The data_names must be aligned with dtypes, in other case it will raise an Error.

    func_parameters: Dict[str, Any], default None
        Extra parameters for the function

    """

    as_data_array = data_names is None
    chunks = [len(coords[dim]) if chunk is None else chunk for chunk, dim in zip(chunks, dims)]
    func_parameters = {} if func_parameters is None else func_parameters

    if as_data_array:
        return _generate_lazy_data_array(
            func=func,
            dims=dims,
            coords=coords,
            chunks=chunks,
            dtype=dtypes,
            func_parameters=func_parameters
        )

    return _generate_lazy_dataset(
        func=func,
        dims=dims,
        coords=coords,
        chunks=chunks,
        dtypes=dtypes,
        func_parameters=func_parameters,
        data_names=data_names
    )


def generate_chunk_coords(coords, chunks, dims) -> Iterable:
    chunk_coords_product = itertools.product(*[
        [coords[dim][i: i + chunk] for i in range(0, len(coords[dim]), chunk)]
        for chunk, dim in zip(chunks, dims)
    ])
    chunk_pos_product = itertools.product(
        *[range(ceil(len(coords[dim]) / chunk)) for chunk, dim in zip(chunks, dims)]
    )
    return zip(chunk_coords_product, chunk_pos_product)


def _generate_lazy_data_array(
        func: Callable,
        dims: List[Hashable],
        coords: Dict[Hashable, Union[List, np.ndarray]],
        chunks: List[int],
        dtype: Any,
        func_parameters: Dict[str, Any],
):
    """
    This method works using the task graph of dask, which is a little complex to use, but it reduces the time
    for generating the array from multiple arrays.

    The important part is that the task graph is a simple dictionary that has as a key the name of the array
    and the chunk position, so every chunk is contiguous to the other, and they follow the order of 0, 1, 2, 3 and so on
    for two-dimensional cases the positions looks like (0, 0), (0, 1), ..., (N, 0), (N, 1), ... (N, M).

    It's important to keep aligned the chunks coords and the chunks positions, the order is maintained due
    to the way that itertools product works.

    """
    task_graph = {}
    name = f'{func.__name__}-{dask.base.tokenize(func.__name__, dims, coords, chunks, dtype, func_parameters)}'

    for chunk_coords, chunk_pos in generate_chunk_coords(coords, chunks, dims):
        chunk_coords = {dim: coord for dim, coord in zip(dims, chunk_coords)}
        parameters = {**func_parameters, **{'coords': chunk_coords}}
        task_graph[(name, *chunk_pos)] = (dask.utils.apply, func, [], parameters)

    arr = dask.array.Array(
        dask=task_graph,
        name=name,
        shape=list(len(coords[dim]) for dim in dims),
        chunks=chunks,
        dtype=dtype
    )

    return xr.DataArray(
        arr,
        dims=dims,
        coords=coords
    )


def _generate_lazy_dataset(
        func: Callable,
        dims: List[Hashable],
        coords: Dict[Hashable, Union[List, np.ndarray]],
        chunks: List[int],
        dtypes: List[Any],
        data_names: List[Hashable],
        func_parameters: Dict[str, Any],
        pure: bool = True
):
    if len(dtypes) != len(data_names):
        raise ValueError(
            f'The number of dtypes ({len(dtypes)}) does not match the number of dataset names ({len(data_names)}), '
            f'you need to specify a dtype for every data array in your dataset'
        )

    dataset_graph = {}
    base_name = f'{func.__name__}-{dask.base.tokenize(func.__name__, dims, coords, chunks, dtypes, func_parameters)}'
    dependencies = []

    for chunk_coords, chunk_pos in generate_chunk_coords(coords, chunks, dims):
        chunk_coords = {dim: coord for dim, coord in zip(dims, chunk_coords)}
        func_parameters['coords'] = chunk_coords
        multi_delayed = dask.delayed(func, nout=len(data_names), pure=pure)(**func_parameters)

        shape = list(len(chunk_coords[dim]) for dim in dims)
        for delayed, data_name, dtype in zip(multi_delayed, data_names, dtypes):
            array = dask.array.from_delayed(delayed, shape=shape, dtype=dtype)
            dataset_graph[(f'{base_name}-{data_name}', *chunk_pos)] = (array.name,) + (0,) * len(dims)
            dependencies.append(array)

    shape = list(len(coords[dim]) for dim in dims)
    return xr.Dataset({
        data_name: xr.DataArray(
            dask.array.Array(
                name=f'{base_name}-{data_name}',
                dask=dask.highlevelgraph.HighLevelGraph.from_collections(
                    f'{base_name}-{data_name}',
                    dataset_graph,
                    dependencies=dependencies
                ),
                chunks=chunks,
                shape=shape,
                dtype=dtype
            ),
            dims=dims,
            coords=coords
        )
        for data_name, dtype in zip(data_names, dtypes)
    })
