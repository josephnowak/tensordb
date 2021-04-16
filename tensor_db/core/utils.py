import xarray
import numpy as np
import pandas as pd

from loguru import logger


def create_dummy_array(n_rows, n_cols, coords=None, dtype=None) -> xarray.DataArray:
    coords = coords
    if coords is None:
        dtype = dtype
        if dtype is None:
            dtype = '<U15'
        coords = {
            'index': np.sort(np.array(list(map(str, range(n_rows))), dtype=dtype)),
            'columns': np.sort(np.array(list(map(str, range(n_cols))), dtype=dtype))
        }

    return xarray.DataArray(
        np.random.rand(n_rows, n_cols),
        dims=['index', 'columns'],
        coords=coords
    )


def create_1d_dummy_array(n, coords=None) -> xarray.DataArray:
    coords = coords
    if coords is None:
        coords = {
            'index': np.sort(np.array(list(map(str, range(n))), dtype='<U15')),
        }

    return xarray.DataArray(
        np.random.rand(n),
        dims=['index'],
        coords=coords
    )


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def modify_coord_dtype(coord: np.array, dtype: str):
    if dtype == 'datetime':
        return pd.to_datetime(coord)

    if dtype == 'int':
        return coord.astype(int)

    if dtype == 'float':
        return coord.astype(float)

    if dtype == 'str':
        return coord.astype(str)

    return coord


def compare_dataset(a, b):
    if isinstance(b, xarray.Dataset):
        b = b.to_array()

    equals = True
    for name, coord in a.coords.items():
        equals &= coord.equals(b.coords[name])

    if isinstance(a, xarray.Dataset):
        a = a.to_array().loc[b.coords]
    equals &= np.allclose(a.values, b.values, equal_nan=True)
    return equals



