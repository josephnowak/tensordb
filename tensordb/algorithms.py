import numpy as np
import xarray as xr
import dask.array as da

from typing import Union, List, Dict, Literal


def ffill(
        arr: xr.DataArray,
        dim: str,
        limit: int = None,
        until_last_valid: Union[xr.DataArray, bool] = False
):
    result = arr
    try:
        result = result.ffill(dim, limit=limit)
    except NotImplementedError:
        from bottleneck import push

        result = xr.DataArray(
            da.apply_along_axis(
                func1d=push,
                axis=result.dims.index(dim),
                arr=arr.data,
                dtype=arr.dtype,
                shape=(arr.sizes[dim], ),
                n=limit,
            ),
            coords=arr.coords,
            dims=arr.dims
        )

    if isinstance(until_last_valid, bool) and until_last_valid:
        until_last_valid = arr.notnull().cumsum(dim=dim).idxmax(dim=dim)

    if isinstance(until_last_valid, xr.DataArray):
        result = result.where(arr.coords[dim] <= until_last_valid, np.nan)

    return result


def rank(
        arr: xr.DataArray,
        dim: str,
        method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'ordinal',
        rank_nan: bool = False
):
    try:
        if method == 'average' and not rank_nan:
            return arr.rank(dim=dim)
        raise NotImplementedError
    except NotImplementedError:
        from scipy.stats import rankdata

        def _nanrankdata_1d(a, method):
            y = np.empty(a.shape, dtype=np.float64)
            y.fill(np.nan)
            idx = ~np.isnan(a)
            y[idx] = rankdata(a[idx], method=method)
            return y

        func = rankdata if rank_nan else _nanrankdata_1d

        return xr.DataArray(
            da.apply_along_axis(
                func1d=func,
                axis=arr.dims.index(dim),
                arr=arr.data,
                dtype=float,
                shape=(arr.sizes[dim],),
            ),
            coords=arr.coords,
            dims=arr.dims
        )



