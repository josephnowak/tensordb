import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import numpy_groupies as npg

from typing import Union, List, Dict, Literal, Any, Tuple
from loguru import logger


class Algorithms:

    @classmethod
    def ffill(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            limit: int = None,
            until_last_valid: Union[xr.DataArray, bool] = False,
    ) -> xr.DataArray:

        result = new_data.ffill(dim=dim, limit=limit)

        if isinstance(until_last_valid, bool) and until_last_valid:
            until_last_valid = new_data.notnull().cumsum(dim=dim).idxmax(dim=dim)

        if isinstance(until_last_valid, (xr.DataArray, xr.Dataset)):
            result = result.where(new_data.coords[dim] <= until_last_valid, np.nan)

        return result

    @classmethod
    def rank(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'ordinal',
            rank_nan: bool = False
    ) -> xr.DataArray:
        try:
            if method == 'average' and not rank_nan:
                return new_data.rank(dim=dim)
            raise NotImplementedError
        except NotImplementedError:

            if isinstance(new_data, xr.Dataset):
                return xr.Dataset(
                    {
                        name: cls.rank(data, dim, method, rank_nan)
                        for name, data in new_data.items()
                    },
                    coords=new_data.coords,
                    attrs=new_data.attrs
                )

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
                dims=new_data.dims,
                attrs=new_data.attrs
            )

    @classmethod
    def shift_on_valid(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            shift: int
    ):
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.shift_on_valid(data, dim, shift)
                    for name, data in new_data.items()
                },
                coords=new_data.coords,
                attrs=new_data.attrs
            )

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
            dims=new_data.dims,
            attrs=new_data.attrs
        )

    @classmethod
    def rolling_along_axis(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            window: int,
            operator: str,
            min_periods: int = None,
            drop_nan: bool = True,
            fill_method: str = None
    ):
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.rolling_along_axis(data, dim, window, operator, min_periods, drop_nan, fill_method)
                    for name, data in new_data.items()
                },
                coords=new_data.coords,
                attrs=new_data.attrs
            )

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
            dims=new_data.dims,
            attrs=new_data.attrs
        )

    @classmethod
    def replace(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            to_replace: Dict,
            dtype: Any = None,
            method: Literal['unique', 'vectorized'] = 'unique',
            default_value: Union[Any, None] = np.nan
    ):
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.replace(data, to_replace, dtype, method, default_value)
                    for name, data in new_data.items()
                },
                coords=new_data.coords,
                attrs=new_data.attrs
            )

        dtype = dtype if dtype else new_data.dtype
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
            dims=new_data.dims,
            attrs=new_data.attrs
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
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.vindex(data, coords)
                    for name, data in new_data.items()
                },
                coords={
                    dim: coords.get(dim, coord) for dim, coord in new_data.coords.items()
                },
                attrs=new_data.attrs
            )

        arr = new_data.data
        equal = True
        for i, dim in enumerate(new_data.dims):
            if dim in coords and not np.array_equal(coords[dim], new_data.coords[dim]):
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
            },
            attrs=new_data.attrs
        )

    @classmethod
    def merge_duplicates_coord(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            func: str,
            fill_value: int = np.nan
    ):
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.merge_duplicates_coord(data, dim, func, fill_value)
                    for name, data in new_data.items()
                },
                coords={
                    dim: coords.get(dim, coord) for dim, coord in new_data.coords.items()
                },
                attrs=new_data.attrs
            )

        # TODO: Delete this method once Xarray merge flox to speed up the groupby and avoid this kind of optimizations
        if new_data.indexes[dim].is_unique:
            return new_data

        axis = new_data.dims.index(dim)
        chunk_axis = new_data.chunks[:axis] + (new_data.sizes[dim],) + new_data.chunks[axis + 1:]
        data = new_data.data.rechunk(chunk_axis)
        unique_coord = pd.Index(np.unique(new_data.indexes[dim]))
        group_idx = unique_coord.get_indexer_for(new_data.coords[dim].values)

        def _reduce(x):
            return npg.aggregate(group_idx, x, axis=axis, func=func, fill_value=fill_value)

        return xr.DataArray(
            data.map_blocks(
                func=_reduce,
                dtype=data.dtype,
                chunks=new_data.chunks[:axis] + (len(unique_coord),) + new_data.chunks[axis + 1:]
            ),
            coords={d: unique_coord if d == dim else v for d, v in new_data.coords.items()},
            dims=new_data.dims,
            attrs=new_data.attrs
        )
