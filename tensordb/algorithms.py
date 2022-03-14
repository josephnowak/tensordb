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
            rank_nan: bool = False,
            use_map_block: bool = False
    ) -> xr.DataArray:
        """
        This is an implementation of scipy rankdata on xarray, with the possibility to avoid the rank of the nans.

        Note:
        there are two implementations of the algorithm, one using dask map_blocks with an axis on the rankdata func
        and the other using dask apply_along_axis without an axis on the rankdata func, I think that the last one
        is better when there are many nans on the data
        """
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

            if use_map_block:
                def _nanrankdata(a, method, axis):
                    return np.where(np.isnan(a), np.nan, rankdata(a, method=method, axis=axis))

                func = rankdata if rank_nan else _nanrankdata
                data = new_data.chunk({dim: None}).data
                ranked = data.map_blocks(
                    func=func,
                    dtype=np.float64,
                    chunks=data.chunks,
                    axis=new_data.dims.index(dim),
                    method=method
                )
            else:
                def _nanrankdata_1d(a, method):
                    idx = ~np.isnan(a)
                    a[idx] = rankdata(a[idx], method=method)
                    return a

                func = rankdata if rank_nan else _nanrankdata_1d
                ranked = da.apply_along_axis(
                    func1d=func,
                    axis=new_data.dims.index(dim),
                    arr=new_data.data,
                    dtype=float,
                    shape=(new_data.sizes[dim],),
                    method=method,
                )

            return xr.DataArray(
                ranked,
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
    ):
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.replace(data, to_replace, dtype)
                    for name, data in new_data.items()
                },
                coords=new_data.coords,
                attrs=new_data.attrs
            )

        dtype = dtype if dtype else new_data.dtype
        sorted_key_groups = np.array(sorted(list(to_replace.keys())))
        group_values = np.array([to_replace[v] for v in sorted_key_groups])

        def _replace(x):
            valid_replace = np.isin(x, sorted_key_groups)
            positions = np.searchsorted(sorted_key_groups, x) * valid_replace
            return np.where(valid_replace, group_values[positions], x)

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
    def apply_on_groups(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            groups: Dict,
            dim: str,
            func: str,
            fill_value: Any = np.nan,
            keep_shape: bool = False
    ):
        """
        Method created as a replacement for the xarray groupby that right now has performance problems,
        this is going to unify the chunks along the dim, so it is memory inefficient.

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            Data on which apply the groupby

        groups: Dict
            The key represent the coords of the dimension and the value the group to which it belongs, this is use
            for create the group_idx of numpy-groupies

        dim: str
            Dimension on which apply the groupby

        func: str
            Function to be applied on the groupby, read numpy-groupies docs

        fill_value: Any
            Read numpy-groupies docs for more info

        keep_shape: bool, default False
            Indicate if the array want to be reduced or not base on the groups, to preserve the shape this algorithm
            is going to replace the original value by its corresponding result of the groupby algorithm
        """
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.apply_on_groups(data, groups, dim, func, fill_value, keep_shape)
                    for name, data in new_data.items()
                },
                attrs=new_data.attrs
            )

        axis = new_data.dims.index(dim)

        unique_groups = pd.Index(np.unique(list(groups.values())))
        group_idx = unique_groups.get_indexer_for([groups[v] for v in new_data.coords[dim].values])
        new_coord = unique_groups
        if keep_shape:
            new_coord = new_data.coords[dim].values

        def _reduce(x):
            arr = npg.aggregate(group_idx, x, axis=axis, func=func, fill_value=fill_value)
            if keep_shape:
                arr = np.take(arr, group_idx, axis=axis)
            return arr

        data = new_data.chunk({dim: None}).data

        return xr.DataArray(
            data.map_blocks(
                func=_reduce,
                dtype=data.dtype,
                chunks=new_data.chunks[:axis] + (len(new_coord),) + new_data.chunks[axis + 1:]
            ),
            coords={d: new_coord if d == dim else v for d, v in new_data.coords.items()},
            dims=new_data.dims,
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
        """
        Group and merge duplicates coord base on a function, this can be a sum or a max. Read numpy-groupies
        docs for more info.
        Internally it calls :meth:`Algorithms.apply_on_groups`
        """
        if new_data.indexes[dim].is_unique:
            return new_data

        return cls.apply_on_groups(
            new_data=new_data,
            groups={v: v for v in new_data.indexes[dim]},
            dim=dim,
            func=func,
            fill_value=fill_value,
            keep_shape=False
        )
