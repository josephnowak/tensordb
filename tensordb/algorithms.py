from typing import Union, List, Dict, Literal, Any, Callable

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from scipy.stats import rankdata
from numba import guvectorize


class NumpyAlgorithms:
    @staticmethod
    def shift_on_valid(a, shift):
        pos = np.arange(len(a))[~np.isnan(a)]
        v = a[np.roll(pos, shift)]
        if shift < 0:
            v[shift:] = np.nan
        else:
            v[:shift] = np.nan
        a[pos] = v
        return a

    @staticmethod
    def apply_rolling_operator(a, drop_nan, window, min_periods, operator, fill_method):
        s = pd.Series(a)
        index = s.index
        if drop_nan:
            s.dropna(inplace=True)
        s = getattr(s.rolling(window, min_periods=min_periods), operator)()
        if drop_nan:
            s = s.reindex(index, method=fill_method)
        return s.values

    @staticmethod
    def replace_unique(x, sorted_key_groups, group_values, default_replace):
        uniques, index = np.unique(x, return_inverse=True)

        valid_replaces = np.isin(sorted_key_groups, uniques, assume_unique=True)
        sorted_key_groups = sorted_key_groups[valid_replaces]
        group_values = group_values[valid_replaces]

        valid_replaces = np.isin(uniques, sorted_key_groups, assume_unique=True)
        uniques[valid_replaces] = group_values

        if default_replace is not None:
            uniques[~valid_replaces] = default_replace

        return uniques[index].reshape(x.shape)

    @staticmethod
    def replace(x, sorted_key_groups, group_values, default_replace):
        if len(sorted_key_groups) == 0:
            if default_replace is not None:
                x = x.copy()
                x[:] = default_replace
            return x

        valid_replace = np.isin(x, sorted_key_groups)
        # put 0 to the positions that are not in the keys of the groups
        positions = np.searchsorted(sorted_key_groups, x) * valid_replace
        if default_replace is not None:
            x = default_replace
        arr = np.where(valid_replace, group_values[positions], x)
        return arr

    @staticmethod
    def cumulative_on_sort(
            x,
            axis,
            cum_func,
            ascending: bool = False,
            keep_nan: bool = True
    ):
        arg_x = x
        # The sort factor allows to modify the order of the sort in numpy
        sort_factor = 1 if ascending else -1

        x = sort_factor * np.sort(sort_factor * x, axis=axis)
        x = cum_func(x, axis=axis)

        # Get the argsort of the original array
        undo_sort = np.argsort(sort_factor * arg_x, axis=axis)
        # Then get the argsort of the argsort which is equivalent to the original order
        undo_sort = np.argsort(undo_sort, axis=axis)

        x = np.take_along_axis(x, undo_sort, axis=axis)
        if keep_nan:
            x[np.isnan(arg_x)] = np.nan
        return x


class Algorithms:
    @classmethod
    def map_blocks_along_axis(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            func,
            dim: str,
            dtype,
            **kwargs
    ) -> xr.DataArray:

        data = new_data.chunk({dim: -1}).data
        return xr.DataArray(
            data.map_blocks(
                dtype=dtype,
                chunks=data.chunks,
                func=func,
                **kwargs
            ),
            coords=new_data.coords,
            dims=new_data.dims,
            attrs=new_data.attrs
        )

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
            ascending=True,
            nan_policy: Literal["omit", "propagate", "error"] = "omit"
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        This is an implementation of scipy rankdata on xarray, with the possibility to avoid the rank of the nans.

        """
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.rank(data, dim, method, ascending, nan_policy)
                    for name, data in new_data.items()
                },
                coords=new_data.coords,
                attrs=new_data.attrs
            )

        if not ascending:
            new_data = -new_data

        def _rank(x, axis, method, nan_policy):
            ranked = rankdata(
                x, method=method, axis=axis, nan_policy=nan_policy
            ).astype(np.float64)
            ranked[np.isnan(x)] = np.nan
            return ranked

        return cls.map_blocks_along_axis(
            new_data,
            func=_rank,
            dtype=np.float64,
            axis=new_data.dims.index(dim),
            method=method,
            nan_policy=nan_policy,
            dim=dim
        )

    @classmethod
    def multi_rank(
            cls,
            new_data: xr.DataArray,
            tie_dim: str,
            dim: str,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Multi rank implemented using the lexsort of numpy, the nan are keep.

        The same dimensions/coords are kept for compatibility with the apply_on_groups
        but all of them are going to contain the same ranking
        """

        def _multi_rank(x, axis, tie_axis):
            shape_tie_axis = x.shape[tie_axis]
            x = np.split(x, shape_tie_axis, axis=tie_axis)[::-1]
            r = np.lexsort(x, axis=axis).argsort(axis=axis) + 1
            r = r.astype(np.float64)
            r[np.isnan(x[-1])] = np.nan
            r = np.concatenate([r] * shape_tie_axis, axis=tie_axis)
            return r

        data = new_data.chunk({dim: -1, tie_dim: -1}).data
        return xr.DataArray(
            data.map_blocks(
                dtype=np.float64,
                chunks=data.chunks,
                func=_multi_rank,
                axis=new_data.dims.index(dim),
                tie_axis=new_data.dims.index(tie_dim),
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

        return xr.DataArray(
            da.apply_along_axis(
                func1d=NumpyAlgorithms.shift_on_valid,
                axis=new_data.dims.index(dim),
                arr=new_data.data,
                shift=shift,
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

        return xr.DataArray(
            da.apply_along_axis(
                func1d=NumpyAlgorithms.apply_rolling_operator,
                axis=new_data.dims.index(dim),
                arr=new_data.data,
                window=window,
                drop_nan=drop_nan,
                min_periods=min_periods,
                operator=operator,
                fill_method=fill_method,
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
            default_replace=None
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

        return xr.DataArray(
            da.map_blocks(
                NumpyAlgorithms.replace,
                new_data.data,
                sorted_key_groups=sorted_key_groups,
                dtype=dtype,
                group_values=group_values,
                chunks=new_data.chunks,
                default_replace=default_replace
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
            func: Union[str, Callable],
            keep_shape: bool = False,
            unique_groups: np.ndarray = None,
            **kwargs
    ):
        """
        This method was created as a replacement of the groupby of Xarray when the group is only
        over one dimension or when the group is of the same shape as the data and the func must be applied
        over a specific dim

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

        func: Union[str, Callable]
            Function to be applied on the groupby, this can be any of the function name that are
            in Xarray (like sum, cumprod and so on) + any function of the Algorithm class

        keep_shape: bool, default False
            Indicate if the array want to be reduced or not base on the groups, to preserve the shape this algorithm
            is going to replace the original value by its corresponding result of the groupby algorithm

        unique_groups: np.ndarray, default None
            Useful when the group array has the same shape as the data and more than one dim, for this case
            is necessary extract the unique elements, so you can provide them here (optional).

        **kwargs
            Any extra parameter to send to the function

        """
        if isinstance(new_data, xr.Dataset):
            return xr.Dataset(
                {
                    name: cls.apply_on_groups(data, groups, dim, func, keep_shape)
                    for name, data in new_data.items()
                },
                attrs=new_data.attrs
            )

        if isinstance(groups, dict):
            groups = xr.DataArray(
                list(groups.values()),
                dims=[dim],
                coords={dim: list(groups.keys())}
            )

        if len(groups.dims) != 1 and groups.dims != new_data.dims:
            raise ValueError(
                f'The dimension of the groups must be the same as the dimension of the new_data '
                f'or it must has only one of its dimensions, '
                f'but got {groups.dims} and {new_data.dims}'
            )

        axis = new_data.dims.index(dim)
        groups.name = "group"

        if unique_groups is None:
            unique_groups = da.unique(groups.data).compute()

        output_coord = new_data.coords[dim].values
        if not keep_shape:
            # In case of grouping by an array of more than 1 dimension and the keep_shape is False.
            output_coord = unique_groups

        chunks = new_data.chunks[:axis] + (len(output_coord),) + new_data.chunks[axis + 1:]

        def _reduce(x, g):
            if len(g.dims) == 1:
                grouped = x.groupby(g)
                if not isinstance(func, str):
                    arr = grouped.map(func, **kwargs)
                elif hasattr(grouped, func):
                    arr = getattr(grouped, func)(dim=dim, **kwargs)
                else:
                    arr = grouped.map(getattr(Algorithms, func), dim=dim, **kwargs).compute()

                if "group" not in arr.dims:
                    # If the function do not reduce the dimension then preserve the same arr
                    pass
                else:
                    arr = arr.rename({"group": dim})
                    if keep_shape:
                        arr = arr.reindex({dim: g.values})
                        arr.coords[dim] = g.coords[dim]
                    else:
                        arr = arr.reindex({dim: unique_groups})

                return arr

            f_dim = next(d for d in x.dims if d != dim)
            arr = xr.concat([
                _reduce(x.sel({f_dim: v}), g.sel({f_dim: v}))
                for v in x.coords[f_dim].values
            ], dim=f_dim)
            arr = arr.transpose(*x.dims)
            return arr

        data = new_data.chunk({dim: -1})
        if len(groups.dims) == len(data.dims):
            groups = groups.chunk(data.chunks)
        else:
            groups = groups.compute()
        new_coords = {k: output_coord if k == dim else v for k, v in new_data.coords.items()}

        data = data.map_blocks(
            _reduce,
            [groups],
            template=xr.DataArray(
                da.empty(
                    dtype=np.float64,
                    chunks=chunks,
                    shape=[len(new_coords[v]) for v in new_data.dims]
                ),
                coords=new_coords,
                dims=new_data.dims,
            )
        )
        return data

    @classmethod
    def merge_duplicates_coord(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            func: str,
    ):
        """
        Group and merge duplicates coord base on a function, this can be a sum or a max. Read numpy-groupies
        docs for more info.
        Internally it calls :meth:`Algorithms.apply_on_groups`
        """
        if new_data.indexes[dim].is_unique:
            return new_data

        new_data = new_data.copy()
        groups = {i: v for i, v in enumerate(new_data.indexes[dim])}
        new_data.coords[dim] = np.arange(new_data.sizes[dim])

        return cls.apply_on_groups(
            new_data=new_data,
            groups=groups,
            dim=dim,
            func=func,
            keep_shape=False
        )

    @classmethod
    def dropna(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dims: List[str],
            how: Literal['all'] = 'all',
            client: Client = None
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
            client: Client = None
    ):
        """
        Equivalent of xarray dropna but for boolean and for multiple dimension and restricted to the all option
        """
        # TODO: Add unit testing
        valid_coords = [
            new_data.any([d for d in new_data.dims if d != dim])
            for dim in dims
        ]
        if client is None:
            valid_coords = dask.compute(*valid_coords)
        else:
            valid_coords = [c.result() for c in client.compute(valid_coords)]
        return new_data.sel(dict(zip(dims, valid_coords)))

    @classmethod
    def append_previous(
            cls,
            old_data: Union[xr.DataArray, xr.Dataset],
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
    ):
        """
        This method only add at the beginning of the new_data the previous data, and this only
        works if the new data is sorted in ascending order over the dimension.

        """
        # Find the nearest coord that is smaller than the first one of the new_data
        position = old_data.indexes[dim][old_data.indexes[dim] < new_data.indexes[dim][0]]
        if len(position) == 0:
            return new_data
        position = position[-1]
        return xr.concat([
            old_data.sel({dim: [position]}).compute(), new_data
        ], dim=dim)

    @classmethod
    def cumulative_on_sort(
            cls,
            new_data: Union[xr.DataArray, xr.Dataset],
            dim: str,
            func: Union[str, Callable],
            ascending=False,
            keep_nan=True
    ):
        """
        Apply cumulative calculation like cumsum and so on but on a sorted way,
        this is useful to generate cumulative rankings

        """
        func = getattr(Algorithms, func) if isinstance(func, str) else func
        return Algorithms.map_blocks_along_axis(
            new_data,
            dtype=new_data.dtype,
            dim=dim,
            func=NumpyAlgorithms.cumulative_on_sort,
            axis=new_data.dims.index(dim),
            cum_func=func,
            keep_nan=keep_nan,
            ascending=ascending,
        )
