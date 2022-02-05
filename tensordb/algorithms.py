import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import string

from typing import Union, List, Dict, Literal, Any, Tuple
from loguru import logger


class Algorithms:

    @classmethod
    def ffill(
            cls,
            new_data: xr.DataArray,
            dim: str,
            limit: int = None,
            until_last_valid: Union[xr.DataArray, bool] = False,
    ) -> xr.DataArray:

        result = new_data.ffill(dim=dim, limit=limit)

        if isinstance(until_last_valid, bool) and until_last_valid:
            until_last_valid = new_data.notnull().cumsum(dim=dim).idxmax(dim=dim)

        if isinstance(until_last_valid, xr.DataArray):
            result = result.where(new_data.coords[dim] <= until_last_valid, np.nan)

        return result

    @classmethod
    def rank(
            cls,
            new_data: xr.DataArray,
            dim: str,
            method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'ordinal',
            rank_nan: bool = False
    ) -> xr.DataArray:
        try:
            if method == 'average' and not rank_nan:
                return new_data.rank(dim=dim)
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
                    axis=new_data.dims.index(dim),
                    arr=new_data.data,
                    dtype=float,
                    shape=(new_data.sizes[dim],),
                ),
                coords=new_data.coords,
                dims=new_data.dims
            )

    @classmethod
    def shift_on_valid(
            cls,
            new_data: xr.DataArray,
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
                axis=new_data.dims.index(dim),
                arr=new_data.data,
                dtype=float,
                shape=(new_data.sizes[dim],),
            ),
            coords=new_data.coords,
            dims=new_data.dims
        )

    @classmethod
    def rolling_along_axis(
            cls,
            new_data: xr.DataArray,
            dim: str,
            window: int,
            operator: str,
            min_periods: int = None,
            drop_nan: bool = True,
            fill_method: str = None
    ):

        def _apply_rolling_operator(a):
            s = pd.Series(a)
            index = s.index
            if drop_nan:
                s.dropna(inplace=True)
            s = getattr(s.rolling(window, min_periods=min_periods), operator)()
            if drop_nan:
                s = s.reindex(index, method=fill_method)
            return s.values

        return xr.DataArray(
            da.apply_along_axis(
                func1d=_apply_rolling_operator,
                axis=new_data.dims.index(dim),
                arr=new_data.data,
                dtype=new_data.dtype,
                shape=(new_data.sizes[dim],),
            ),
            coords=new_data.coords,
            dims=new_data.dims
        )

    @classmethod
    def replace(
            cls,
            new_data: xr.DataArray,
            to_replace: Dict,
            dtype: Any = None,
            method: Literal['unique', 'vectorized'] = 'unique',
            default_value: Union[Any, None] = np.nan
    ):
        dtype = dtype if dtype else new_data.dtype
        to_replace = pd.Series(to_replace).drop_duplicates().sort_index()
        vectorized_map = np.vectorize(
            lambda e: to_replace.get(e, e if default_value is None else default_value),
            otypes=[dtype],
            signature='()->()'
        )

        if method == 'vectorized':
            _replace = vectorized_map
        elif method == 'unique':
            def _replace(x):
                unique_elements, rebuild_index = np.unique(x, return_inverse=True)
                return vectorized_map(unique_elements)[rebuild_index].reshape(x.shape)
        else:
            raise NotImplemented(f'The method {method} is not implemented')

        return xr.DataArray(
            da.map_blocks(
                _replace,
                new_data.data,
                dtype=dtype,
                chunks=new_data.chunks
            ),
            coords=new_data.coords,
            dims=new_data.dims
        )

    @classmethod
    def vindex(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            coords: Dict,
    ):
        """
        Implementation of dask vindex using xarray
        """
        arr = new_data.data
        equal = True
        for i, dim in enumerate(new_data.dims):
            if dim in coords and not np.array_equal(coords[dim], new_data.coords[dim]):
                if not new_data.indexes[dim].is_unique:
                    raise xr.errors.DuplicateLabelError(
                        f'vindex require unique coords on the new_data, delete the duplicates on the dim {dim}'
                    )
                int_coord = new_data.indexes[dim].get_indexer(coords[dim])
                data_slices = (slice(None),) * i + (int_coord,) + (slice(None),) * (len(new_data.dims) - i - 1)
                arr = da.moveaxis(arr.vindex[data_slices], 0, i)
                equal = False

        if equal:
            return new_data

        return xr.DataArray(
            arr,
            dims=new_data.dims,
            coords={
                dim: coords.get(dim, coord) for dim, coord in new_data.coords.items()
            }
        )

    @classmethod
    def merge_duplicates_coord(
            cls,
            new_data,
            dim: str,
            method: str = 'max'
    ):
        # TODO: Delete this once Xarray merge flox to speed up the groupby and avoid this kind of optimizations
        duplicated = np.unique(new_data.indexes[dim][new_data.indexes[dim].duplicated()])
        if len(duplicated) > 0:
            valid = new_data.coords[dim].isin(duplicated)
            duplicate_data = getattr(new_data.sel({dim: valid}).groupby(dim), method)(dim)
            tmp_data = xr.concat([duplicate_data, new_data.sel({dim: ~valid})], dim)
            new_data = cls.vindex(tmp_data, {dim: np.unique(new_data.indexes[dim])})

        return new_data

