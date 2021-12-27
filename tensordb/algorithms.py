import numpy as np
import xarray as xr
import dask.array as da

from typing import Union, List, Dict, Literal


def ffill(
        arr: xr.DataArray,
        dim: str,
        limit: int = None,
        until_last_valid: Union[xr.DataArray, bool] = False,
):
    from bottleneck import push

    # TODO delete the forward fill logic once I solve https://github.com/pydata/xarray/issues/6112

    def _fill_with_last_one(a, b):
        # cumreduction apply the push func over all the blocks first so,
        # the only missing part is filling the missing values using
        # the last data for every one of them
        if isinstance(a, np.ma.masked_array) or isinstance(b, np.ma.masked_array):
            a = np.ma.getdata(a)
            b = np.ma.getdata(b)
            values = np.where(~np.isnan(b), b, a)
            return np.ma.masked_array(values, mask=np.ma.getmaskarray(b))

        return np.where(~np.isnan(b), b, a)

    def _ffill(x):
        return xr.DataArray(
            da.reductions.cumreduction(
                func=push,
                binop=_fill_with_last_one,
                ident=np.nan,
                x=x.data,
                axis=x.dims.index(dim),
                dtype=x.dtype,
                method="sequential",
            ),
            dims=x.dims,
            coords=x.coords
        )

    result = _ffill(arr)
    if limit is not None:
        axis = arr.dims.index(dim)
        arange = xr.DataArray(
            da.broadcast_to(
                da.arange(
                    arr.shape[axis],
                    chunks=arr.chunks[axis],
                    dtype=arr.dtype
                ).reshape(
                    tuple(size if i == axis else 1 for i, size in enumerate(arr.shape))
                ),
                arr.shape,
                arr.chunks
            ),
            coords=arr.coords,
            dims=arr.dims
        )
        valid_limits = (arange - _ffill(arange.where(arr.notnull(), np.nan))) <= limit
        result = result.where(valid_limits, np.nan)

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


def shift_on_valid(
        arr: xr.DataArray,
        dim: str,
        shift: int
):
    def _shift(a):
        pos = np.arange(len(a))[~np.isnan(a)]
        v = a[np.roll(pos, shift)]
        if shift < 0:
            v[shift:] = np.nan
        else:
            v[:shift] = np.nan
        a[pos] = v
        return a

    return xr.DataArray(
        da.apply_along_axis(
            func1d=_shift,
            axis=arr.dims.index(dim),
            arr=arr.data,
            dtype=float,
            shape=(arr.sizes[dim],),
        ),
        coords=arr.coords,
        dims=arr.dims
    )
