import xarray as xr
import dask
import numpy as np
import itertools

from typing import Iterable, Callable, Union, Dict, Any, List, Tuple, Optional, Literal, Hashable
from pydantic import validate_arguments
from loguru import logger


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def defined_translation(
        func: Callable,
        dims: List[Hashable],
        coords: Dict[Hashable, Union[List, np.ndarray]],
        chunks: List[int],
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

    as_data_array = False
    if data_names is None:
        data_names = ['null']
        as_data_array = True

    if not isinstance(dtypes, list):
        dtypes = [dtypes]

    if len(dtypes) != len(data_names):
        raise ValueError(
            f'The number of dtypes ({len(dtypes)}) does not match the number of dataset names ({len(data_names)}), '
            f'you need to specify a dtype for every data array in your dataset'
        )

    total_chunk_coords = []
    for chunk, dim in zip(chunks, dims):
        coord = coords[dim]
        chunk = len(coord) if chunk is None else chunk
        total_chunk_coords.append([coord[i: i + chunk] for i in range(0, len(coord), chunk)])

    func_parameters = {} if func_parameters is None else func_parameters

    chunked_arrays = []
    for chunk_coords in itertools.product(*total_chunk_coords):
        chunk_coords = {dim: coord for dim, coord in zip(dims, chunk_coords)}
        func_parameters['coords'] = chunk_coords
        shape = list(len(chunk_coords[dim]) for dim in dims)
        if as_data_array:
            delayed_func = [dask.delayed(func)(**func_parameters)]
        else:
            delayed_func = dask.delayed(func, nout=len(data_names))(**func_parameters)
        dataset = xr.Dataset({
            name: xr.DataArray(
                dask.array.from_delayed(
                    delayed,
                    shape=shape,
                    dtype=dtype
                ),
                dims=dims,
                coords=chunk_coords
            )
            for delayed, name, dtype in zip(delayed_func, data_names, dtypes)
        })
        chunked_arrays.append(dataset)

    dataset = xr.combine_by_coords(chunked_arrays)
    if as_data_array:
        return dataset[data_names[0]]
    return dataset
