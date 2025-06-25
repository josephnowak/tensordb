from collections.abc import Callable
from typing import Any, Literal

import bottleneck as bn
import dask
import dask.array as da
import numbagg as nba
import numpy as np
import xarray as xr
from dask.distributed import Client
from scipy.stats import rankdata


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
    def apply_rolling_operator(
        x,
        drop_nan,
        window,
        min_periods,
        operator,
        fill_method: Literal["ffill", None],
        inplace=False,
    ):
        min_periods = window if min_periods is None else min_periods

        if not inplace:
            x = x.copy()

        if drop_nan:
            bitmask = ~np.isnan(x)
        else:
            bitmask = np.full(x.shape, True, dtype=bool)

        filter_x = x[bitmask]
        window = min(len(filter_x), window)

        if window < min_periods:
            return np.full(x.shape, np.nan, dtype=x.dtype)

        func = getattr(bn, f"move_{operator}")
        x[bitmask] = func(filter_x, window, min_count=min_periods)

        if drop_nan and fill_method == "ffill":
            x = bn.push(x)

        return x

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
        x, axis, cum_func, ascending: bool = False, keep_nan: bool = True
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

    @staticmethod
    def rank(x, axis, method, nan_policy, ascending=True, use_bottleneck=False):
        if not ascending:
            x = -x

        if use_bottleneck:
            ranked = bn.nanrankdata(x, axis=axis).astype(np.float64)
        else:
            ranked = rankdata(
                x, method=method, axis=axis, nan_policy=nan_policy
            ).astype(np.float64)
        ranked[np.isnan(x)] = np.nan
        return ranked

    @staticmethod
    def multi_rank(x, axis, tie_axis):
        shape_tie_axis = x.shape[tie_axis]
        x = np.split(x, shape_tie_axis, axis=tie_axis)[::-1]
        r = np.lexsort(x, axis=axis).argsort(axis=axis) + 1
        r = r.astype(np.float64)
        r[np.isnan(x[-1])] = np.nan
        r = np.concatenate([r] * shape_tie_axis, axis=tie_axis)
        return r


class Algorithms:
    @classmethod
    def map_blocks_along_axis(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        func,
        dim: str,
        dtype,
        drop_dim: bool = False,
        kwargs: dict = None,
    ) -> xr.DataArray:
        kwargs = kwargs or {}
        template = new_data.chunk({dim: -1})
        data = template.data

        drop_axis = None
        if drop_dim:
            drop_axis = new_data.dims.index(dim)
            template = template.isel({dim: 0}, drop=True)

        chunks = template.chunks

        return xr.DataArray(
            data.map_blocks(
                dtype=dtype, drop_axis=drop_axis, func=func, chunks=chunks, **kwargs
            ),
            coords=template.coords,
            dims=template.dims,
            attrs=new_data.attrs,
        )

    @classmethod
    def ffill(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        limit: int = None,
        until_last_valid: xr.DataArray | bool = False,
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
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        method: Literal["average", "min", "max", "dense", "ordinal"] = "ordinal",
        ascending=True,
        nan_policy: Literal["omit", "propagate", "error"] = "omit",
        use_bottleneck: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        This is an implementation of scipy rankdata on xarray, with the possibility to avoid the rank of the nans.

        If the bottleneck option is enable then the method and nan policy parameters are ignored
        """
        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.rank,
                method=method,
                dim=dim,
                ascending=ascending,
                nan_policy=nan_policy,
                use_bottleneck=use_bottleneck,
            )

        return cls.map_blocks_along_axis(
            new_data,
            func=NumpyAlgorithms.rank,
            dtype=np.float64,
            dim=dim,
            kwargs=dict(
                axis=new_data.dims.index(dim),
                method=method,
                ascending=ascending,
                nan_policy=nan_policy,
                use_bottleneck=use_bottleneck,
            ),
        )

    @classmethod
    def multi_rank(
        cls,
        new_data: xr.DataArray,
        tie_dim: str,
        dim: str,
    ) -> xr.DataArray | xr.Dataset:
        """
        Multi rank implemented using the lexsort of numpy, the nan are keep.

        The same dimensions/coords are kept for compatibility with the apply_on_groups
        but all of them are going to contain the same ranking
        """

        data = new_data.chunk({dim: -1, tie_dim: -1}).data
        return xr.DataArray(
            data.map_blocks(
                dtype=np.float64,
                chunks=data.chunks,
                func=NumpyAlgorithms.multi_rank,
                axis=new_data.dims.index(dim),
                tie_axis=new_data.dims.index(tie_dim),
            ),
            coords=new_data.coords,
            dims=new_data.dims,
            attrs=new_data.attrs,
        )

    @classmethod
    def shift_on_valid(
        cls, new_data: xr.DataArray | xr.Dataset, dim: str, shift: int
    ):
        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.shift_on_valid,
                shift=shift,
                dim=dim,
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
            attrs=new_data.attrs,
        )

    @classmethod
    def rolling_along_axis(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        window: int,
        operator: str,
        min_periods: int = None,
        drop_nan: bool = True,
        fill_method: str = None,
    ):
        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.rolling_along_axis,
                window=window,
                dim=dim,
                operator=operator,
                min_periods=min_periods,
                drop_nan=drop_nan,
                fill_method=fill_method,
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
                inplace=True,
            ),
            coords=new_data.coords,
            dims=new_data.dims,
            attrs=new_data.attrs,
        )

    @classmethod
    def replace(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        to_replace: dict,
        dtype: Any = None,
        default_replace=None,
    ):
        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.replace,
                to_replace=to_replace,
                dtype=dtype,
                default_replace=default_replace,
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
                default_replace=default_replace,
            ),
            coords=new_data.coords,
            dims=new_data.dims,
            attrs=new_data.attrs,
        )

    @classmethod
    def vindex(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        coords: dict,
    ):
        """
        Implementation of dask vindex using xarray
        """
        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.vindex,
                coords=coords,
            )

        arr = new_data.data
        equal = True
        for i, dim in enumerate(new_data.dims):
            if dim in coords and not np.array_equal(coords[dim], new_data.coords[dim]):
                int_coord = new_data.indexes[dim].get_indexer(coords[dim])
                data_slices = (
                    (slice(None),) * i
                    + (int_coord,)
                    + (slice(None),) * (len(new_data.dims) - i - 1)
                )
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
            attrs=new_data.attrs,
        )

    @classmethod
    def apply_on_groups(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        groups: dict | xr.DataArray,
        dim: str,
        func: str | Callable,
        keep_shape: bool = False,
        unique_groups: np.ndarray = None,
        kwargs: dict[str, Any] = None,
        template: xr.DataArray | xr.Dataset | str = None,
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

        template: Union[xr.DataArray, xr.Dataset]: str = None
            If the template is not set then is going to be generated internally based on the keep_shape
            parameter and the data vars inside the template (if a Dataset).
            If a string is set then the template is going to be generated internally but based
            on the var name specified

        kwargs: Dict[str, Any] = None,
            Any extra parameter to send to the function

        """
        kwargs = kwargs or dict()

        if isinstance(groups, dict):
            groups = xr.DataArray(
                list(groups.values()), dims=[dim], coords={dim: list(groups.keys())}
            )

        if len(groups.dims) != 1 and groups.dims != new_data.dims:
            raise ValueError(
                f"The dimension of the groups must be the same as the dimension of the new_data "
                f"or it must has only one of its dimensions, "
                f"but got {groups.dims} and {new_data.dims}"
            )

        groups.name = "group"

        if len(groups.dims) != len(new_data.dims):
            groups = groups.compute()

        if unique_groups is None:
            unique_groups = da.unique(groups.data).compute()

        output_coord = new_data.coords[dim].to_numpy()
        if not keep_shape:
            # In case of grouping by an array of more than 1 dimension and the keep_shape is False.
            output_coord = unique_groups

        def generate_template(x):
            axis = x.dims.index(dim)

            chunks = x.chunks[:axis] + (len(output_coord),) + x.chunks[axis + 1 :]
            new_coords = {
                k: output_coord if k == dim else v for k, v in x.coords.items()
            }
            return xr.DataArray(
                da.empty(
                    dtype=np.float64,
                    chunks=chunks,
                    shape=[len(new_coords[v]) for v in x.dims],
                ),
                coords=new_coords,
                dims=x.dims,
            )

        if template is None or isinstance(template, str):
            if isinstance(new_data, xr.Dataset):
                var_name = template
                template = new_data.map(generate_template)
                if isinstance(var_name, str):
                    template = template[var_name]
            else:
                template = generate_template(new_data)

        def _reduce(x, g, func, **kwargs):
            if len(g.dims) == 1:
                grouped = x.groupby(g)
                if not isinstance(func, str):
                    arr = grouped.map(func, **kwargs)
                elif hasattr(grouped, func):
                    arr = getattr(grouped, func)(dim=dim, **kwargs)
                else:
                    arr = grouped.map(
                        lambda v: getattr(Algorithms, func)(
                            v, dim=dim, **kwargs
                        ).compute(scheduler="single-threaded")
                    )

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
                arr = arr.transpose(*x.dims)

                return arr

            f_dim = next(d for d in x.dims if d != dim)
            arr = xr.concat(
                [
                    _reduce(x.sel({f_dim: v}), g.sel({f_dim: v}), func, **kwargs)
                    for v in x.coords[f_dim].values
                ],
                dim=f_dim,
            )
            arr = arr.transpose(*x.dims)
            return arr

        data = new_data.chunk({dim: -1})
        if len(groups.dims) == len(data.dims):
            groups = groups.chunk(data.chunksizes)

        data = data.map_blocks(
            _reduce,
            [groups, func],
            kwargs=kwargs,
            template=template,
        )
        return data

    @classmethod
    def merge_duplicates_coord(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        func: str,
        kwargs: dict[str, Any] = None,
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
            keep_shape=False,
            kwargs=kwargs,
        )

    @classmethod
    def dropna(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dims: list[str],
        how: Literal["all"] = "all",
        client: Client = None,
    ):
        """
        Equivalent of xarray dropna but for multiple dimension and restricted to the all option
        """
        # TODO: Add unit testing
        dropped_data = cls.drop_unmarked(new_data.notnull(), dims=dims, client=client)
        return new_data.sel(dropped_data.coords)

    @classmethod
    def drop_unmarked(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dims: list[str],
        how: Literal["all"] = "all",
        client: Client = None,
    ):
        """
        Equivalent of xarray dropna but for boolean and for multiple dimension and restricted to the all option
        """
        # TODO: Add unit testing
        valid_coords = [
            new_data.any([d for d in new_data.dims if d != dim]) for dim in dims
        ]
        if client is None:
            valid_coords = dask.compute(*valid_coords)
        else:
            valid_coords = [c.result() for c in client.compute(valid_coords)]
        return new_data.sel(dict(zip(dims, valid_coords, strict=False)))

    @classmethod
    def append_previous(
        cls,
        old_data: xr.DataArray | xr.Dataset,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
    ):
        """
        This method only add at the beginning of the new_data the previous data, and this only
        works if the new data is sorted in ascending order over the dimension.

        """
        # Find the nearest coord that is smaller than the first one of the new_data
        position = old_data.indexes[dim][
            old_data.indexes[dim] < new_data.indexes[dim][0]
        ]
        if len(position) == 0:
            return new_data
        position = position[-1]
        return xr.concat([old_data.sel({dim: [position]}).compute(), new_data], dim=dim)

    @classmethod
    def cumulative_on_sort(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        func: str | Callable,
        ascending=False,
        keep_nan=True,
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
            kwargs=dict(
                axis=new_data.dims.index(dim),
                cum_func=func,
                keep_nan=keep_nan,
                ascending=ascending,
            ),
        )

    @classmethod
    def bitmask_topk(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        dim: str,
        top_size,
        tie_breaker_dim: str = None,
    ):
        """
        Create a bitmask where the True values means that the cell is on the top N on the dim.

        This is equivalent to:
        arr >= np.take(arr.topk(top_size, axis=axis), -1, axis=axis)

        The algorithm is implemented using the topk algorithm of Dask, and it automatically fills NaNs
        using -INF before calling the method, which avoid issues related to having the nan in the first
        positions or having nan in the last position because there was no sufficient data.

        The tie_breaker_dim creates by default a structured array that is latter use
        on the topk algorithm of Dask, the only issue right now is that the structure indexes
        do not support inequality operators which is really strange due that they support sort, so
        I have to implement a manual inequality using map blocks which can add some overhead
        """

        top_data = new_data.fillna(-np.inf)
        first_level = new_data
        if tie_breaker_dim is not None:
            from numpy.lib import recfunctions as rfn

            first_level = new_data.isel({tie_breaker_dim: 0}, drop=True)
            top_data = Algorithms.map_blocks_along_axis(
                top_data,
                func=lambda x, axis: rfn.unstructured_to_structured(
                    np.moveaxis(x, axis, -1),
                    [(f"f{i}", x.dtype) for i in range(x.shape[axis])],
                ),
                dim=tie_breaker_dim,
                dtype=[
                    (f"f{i}", new_data.dtype)
                    for i in range(new_data.sizes[tie_breaker_dim])
                ],
                kwargs=dict(axis=new_data.dims.index(tie_breaker_dim)),
                drop_dim=True,
            )

        if top_size >= new_data.sizes[dim]:
            return new_data.notnull()

        if top_size == 0:
            return xr.zeros_like(new_data, dtype=bool)

        axis = top_data.dims.index(dim)

        topk = np.take(top_data.data.topk(top_size, axis=axis), -1, axis=axis)
        topk = xr.DataArray(
            topk,
            dims=[d for d in top_data.dims if d != dim],
            coords={d: v for d, v in top_data.coords.items() if d != dim},
        )

        if tie_breaker_dim is None:
            return first_level >= topk

        def split_structured(x):
            raw = x.values.view(new_data.dtype).reshape(x.shape + (-1,))
            return [
                xr.DataArray(np.take(raw, i, axis=-1), coords=x.coords, dims=x.dims)
                for i in range(raw.shape[-1])
            ]

        def structured_inequality(x, top):
            x = split_structured(x)
            top = split_structured(top)

            bitmask = x[0] > top[0]
            ties = x[0] == top[0]
            for i in range(1, len(x)):
                bitmask |= ties & (x[i] > top[i])
                ties &= x[i] == top[i]

            return bitmask | ties

        bitmask = (
            top_data.map_blocks(
                structured_inequality, args=(topk,), template=first_level.notnull()
            )
            & first_level.notnull()
        )

        return bitmask.expand_dims({tie_breaker_dim: new_data.coords[tie_breaker_dim]})

    @classmethod
    def rolling_overlap(
        cls,
        new_data: xr.DataArray | xr.Dataset,
        func: Callable,
        dim: str,
        window: int,
        window_margin: int,
        min_periods: int = None,
        apply_ffill: bool = True,
        validate_window_size: bool = True,
    ):
        assert window_margin >= window

        if validate_window_size:
            window_margin = min(window_margin, new_data.sizes[dim])

        assert window > 0

        if isinstance(new_data, xr.Dataset):
            return new_data.map(
                cls.rolling_overlap,
                func=func,
                window=window,
                dim=dim,
                window_margin=window_margin,
                min_periods=min_periods,
            )

        min_periods = window if min_periods is None else min_periods
        axis = new_data.dims.index(dim)
        data = new_data.data

        def _apply_on_valid(x):
            bitmask = ~np.isnan(x)
            filter_x = x[bitmask]

            min_window = window
            if validate_window_size:
                min_window = min(window, len(filter_x))
                if min_window == 0:
                    return x

            func(filter_x, window=min_window, min_count=min_periods, out=filter_x)

            x[bitmask] = filter_x
            if apply_ffill:
                nba.ffill(arr=x, limit=len(x), out=x)
            return x

        def apply_on_valid(x):
            return np.apply_along_axis(func1d=_apply_on_valid, axis=axis, arr=x)

        data = data.map_overlap(
            apply_on_valid,
            depth={axis: window_margin},
            boundary=None,
            meta=data,
            trim=True,
        )

        data = xr.DataArray(data, dims=new_data.dims, coords=new_data.coords)
        return data

    @classmethod
    def reindex_with_pad(
        cls,
        data: xr.DataArray | xr.Dataset,
        coords,
        preferred_chunks: dict[str, int],
        fill_value: Any,
        method: str = None,
        apply_chunk: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """
        This function is going to pad artificial data on the dimensions whose
        size is smaller than the preferred chunk before reindexing
        to reduce the amount of chunks, to achieve that it generates an
        autoincrement index that replace the coord
        Note: If it is not necessary to pad data then the algorithm is going to
        use reindex directly and then chunk the data.

        :param data:
            Data that wants to be reindexed
        :param coords:
            Coords to reindex
        :param preferred_chunks:
            Ideal chunk size of the array, if the array is smaller than the chunk
            then some artificial data is added with the pad method to latter be rechunked into a
            single chunk.
            It is important to highlight that if the size of the data on a dim is bigger
            than the preferred chunk size then on that dim the original chunks are going
            to be preserved unless the apply_chunk is set to True (default)
        :param fill_value:
            Used in the reindex method
        :param method
            Used in the reindex method
        :param apply_chunk, default True
            If true then the chunk method is going to be applied after calling the reindex
            method of Xarray (which is called after padding the data).
            Disable this option is useful to properly visualize
            the chunks created by the algorithm and see the reduction in terms of tasks
            and also some other specific use cases
        :return:
            A chunked reindexed data
        """
        if method == "bfill":
            raise NotImplementedError(
                "The bfill method is not implemented for the moment"
            )

        is_coord_empty = False
        reindex_data = data.copy()
        reindex_mapped_pad_coords = {}
        reindex_mapped_coords = {}
        inv_autoincrement_map = {}
        pad_widths = {}

        for dim, coord in coords.items():
            coord = np.array(coord)
            pad_width = preferred_chunks.get(dim, 0) - data.sizes[dim]
            reindex_mapped_coords[dim] = coord
            if pad_width <= 0:
                continue
            data_coord = data.coords[dim].to_numpy()
            is_coord_empty |= len(data_coord) == 0
            total_coord = np.union1d(np.array(coord), data_coord)

            auto_map = {c: i for i, c in enumerate(total_coord)}
            mapped_data_coord = np.array([auto_map[c] for c in data_coord])
            reindex_mapped_coords[dim] = np.array([auto_map[c] for c in coord])
            reindex_mapped_pad_coords[dim] = np.concatenate(
                (
                    mapped_data_coord,
                    list(range(len(total_coord), len(total_coord) + pad_width)),
                ),
            )

            reindex_data.coords[dim] = mapped_data_coord
            inv_autoincrement_map[dim] = {v: k for k, v in auto_map.items()}
            pad_widths[dim] = (0, pad_width)

        if not reindex_mapped_pad_coords or is_coord_empty:
            data = data.reindex(coords, fill_value=fill_value, method=method)
            if apply_chunk:
                data = data.chunk(preferred_chunks)
            return data

        reindex_data = reindex_data.pad(pad_widths, mode="edge")
        for dim, coord in reindex_mapped_pad_coords.items():
            reindex_data.coords[dim] = coord
        reindex_data = reindex_data.chunk(
            {dim: -1 for dim in reindex_mapped_pad_coords.keys()}
        )
        reindex_data = reindex_data.reindex(
            reindex_mapped_coords, method=method, fill_value=fill_value
        )

        for dim in reindex_mapped_pad_coords.keys():
            reindex_data.coords[dim] = np.array(
                [
                    inv_autoincrement_map[dim][c]
                    for c in reindex_data.coords[dim].to_numpy()
                ]
            )

        if apply_chunk:
            reindex_data = reindex_data.chunk(preferred_chunks)
        return reindex_data
