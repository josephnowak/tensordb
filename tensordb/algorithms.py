import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask
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
            # put 0 to the positions that are not in the keys of the groups
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
            groups: Union[Dict, xr.DataArray],
            dim: str,
            func: str,
            fill_value: Any = np.nan,
            keep_shape: bool = False,
            group_algorithm: Literal['aggregate', 'aggregate_nb', 'aggregate_weave'] = 'aggregate',
    ):
        """
        Method created as a replacement for the xarray groupby that right now has performance problems,
        this is going to unify the chunks along the dim, so it is memory inefficient.

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            Data on which apply the groupby

        groups: Dict, xr.DataArray
            The key represent the coords of the dimension and the value the group to which it belongs, this is use
            for create the group_idx of numpy-groupies.

            For the xr.DataArray it must have the same dimensions and coords, and every cell represent the
            corresponding group for the new_data parameter, for this case the resulting array has the same
            shape that new_data, so it always use the keep_shape as True

        dim: str
            Dimension on which apply the groupby

        func: str
            Function to be applied on the groupby, read numpy-groupies docs

        fill_value: Any
            Read numpy-groupies docs for more info

        keep_shape: bool, default False
            Indicate if the array want to be reduced or not base on the groups, to preserve the shape this algorithm
            is going to replace the original value by its corresponding result of the groupby algorithm

        group_algorithm: str, default 'aggregate'
            Algorithm use by numpy_groupies to apply the groupby, read numpy-groupies docs for more info
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
        group_algorithm = getattr(npg, group_algorithm)

        if isinstance(groups, xr.DataArray):
            def _reduce(arr, grouper):
                return np.moveaxis(np.array([
                    group_algorithm(group_idx, x, func=func, fill_value=fill_value)[group_idx]
                    for x, group_idx in zip(np.moveaxis(arr, axis, -1), np.moveaxis(grouper, axis, -1))
                ]), -1, axis)

            data = new_data.chunk({dim: None}).data
            groups = groups.chunk(data.chunks).data

            return xr.DataArray(
                dask.array.map_blocks(
                    _reduce,
                    data,
                    groups,
                    chunks=data.chunks,
                    dtype=float,
                ),
                coords=new_data.coords,
                dims=new_data.dims,
                attrs=new_data.attrs
            )

        unique_groups = pd.Index(np.unique(list(groups.values())))
        group_idx = unique_groups.get_indexer_for([groups[v] for v in new_data.coords[dim].values])
        new_coord = unique_groups
        if keep_shape:
            new_coord = new_data.coords[dim].values

        def _reduce(x):
            arr = group_algorithm(group_idx, x, axis=axis, func=func, fill_value=fill_value)
            arr = np.take(arr, group_idx, axis=axis) if keep_shape else arr
            return arr

        data = new_data.chunk({dim: None}).data

        return xr.DataArray(
            data.map_blocks(
                func=_reduce,
                dtype=data.dtype,
                chunks=new_data.chunks[:axis] + (len(new_coord),) + new_data.chunks[axis + 1:],
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

    @classmethod
    def dropna(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dims: List[str],
            how: Literal['all'] = 'all',
            client: dask.distributed.Client = None
    ):
        """
        Equivalent of xarray dropna but for multiple dimension and restricted to the all option
        """
        # TODO: Add unit testing
        dropped_data = cls.drop_unmarked(
            new_data.notnull(),
            dims=dims,
            client=client
        )
        return new_data.sel(dropped_data.coords)

    @classmethod
    def drop_unmarked(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dims: List[str],
            how: Literal['all'] = 'all',
            client: dask.distributed.Client = None
    ):
        """
        Equivalent of xarray dropna but for boolean and for multiple dimension and restricted to the all option
        """
        # TODO: Add unit testing
        client = dask if client is None else client
        valid_coords = client.compute(*[
            new_data.any([d for d in new_data.dims if d != dim])
            for dim in dims
        ])
        return new_data.sel(dict(zip(dims, valid_coords)))

    @classmethod
    def insert_sorted(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            other: Union[xr.DataArray, xr.DataArray],
            dim: str,
            fill_value: Any = np.nan
    ):
        """
        Insert the data of one array into another. this is equivalent to the combine first method of xarray
        but only applied on one dimension and on a sorted coord, this is to avoid performance
        problem on the reindex

        Parameters
        ----------
        new_data: Union[xr.DataArray, xr.Dataset]
            This is the data at which the data is going to be inserted

        other: Union[xr.DataArray, xr.Dataset]
            Data that is going to be inserted on the new_data array

        fill_value: Any, default np.nan
            Useful to avoid the lost of the original dtype when the reindex is executed

        dim: str
            Dim for the insertion

        """
        # TODO: Add unit testing
        act_coord = new_data.indexes[dim]
        coord_insert = other.indexes[dim]
        if not (act_coord.is_monotonic_increasing or act_coord.is_monotonic_decreasing):
            raise ValueError(f'The new_data coord must be sorted')

        reindex_coord = act_coord.union(coord_insert).sort_values(ascending=act_coord.is_monotonic_increasing)

        new_data = new_data.reindex({dim: reindex_coord.values}, fill_value=fill_value)
        indexing = tuple(coord_insert if d == dim else slice(None, None) for d in new_data.dims)
        new_data.loc[indexing] = other
        return new_data
