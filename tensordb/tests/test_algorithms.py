import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tensordb.algorithms import Algorithms


# TODO: Add more tests for the dataset cases


def test_ffill():
    arr = xr.DataArray(
        [
            [1, np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, np.nan, np.nan, 2, np.nan, np.nan],
            [np.nan, 5, np.nan, 2, np.nan, np.nan],
        ],
        dims=['a', 'b'],
        coords={'a': list(range(3)), 'b': list(range(6))}
    ).chunk(
        (1, 2)
    )
    assert Algorithms.ffill(arr, limit=2, dim='b').equals(arr.compute().ffill('b', limit=2))
    assert Algorithms.ffill(arr, limit=2, dim='b', until_last_valid=True).equals(
        xr.DataArray(
            [
                [1, np.nan, np.nan, np.nan, np.nan, np.nan],
                [1, 1, 1, 2, np.nan, np.nan],
                [np.nan, 5, 5, 2, np.nan, np.nan],
            ],
            dims=['a', 'b'],
            coords={'a': list(range(3)), 'b': list(range(6))}
        )
    )


@pytest.mark.parametrize(
    'method, ascending',
    [
        ('average', True),
        ('min', False),
        ('max', False),
        ('ordinal', True),
        ('dense', False)
    ]
)
def test_rank(method, ascending):
    arr = xr.DataArray(
        [
            [1, 2, 3],
            [4, 4, 1],
            [5, 2, 3],
            [np.nan, 3, 0],
            [8, 7, 9]
        ],
        dims=['a', 'b'],
        coords={'a': list(range(5)), 'b': list(range(3))}
    ).chunk((3, 1))
    df = pd.DataFrame(arr.values, index=arr.a.values, columns=arr.b.values)
    result = Algorithms.rank(
        arr,
        'b',
        method=method,
        ascending=ascending
    )
    rank_pandas_method = 'first' if method == 'ordinal' else method
    expected = df.rank(axis=1, method=rank_pandas_method, na_option='keep', ascending=ascending)
    assert result.equals(
        xr.DataArray(
            expected.values,
            dims=arr.dims,
            coords=arr.coords
        )
    )


@pytest.mark.parametrize(
    'dim',
    ["b", "a"]
)
def test_multi_rank(dim):
    coords = {'a': list(range(5)), 'b': list(range(3)), 'c': list(range(2))}
    arr = xr.DataArray(
        [
            [
                [np.nan, 2, 2],
                [4, 4, 1],
                [5, 2, 2],
                [np.nan, 0, 3],
                [7, np.nan, 9]
            ],
            [
                [3, np.nan, 3],
                [3, 3, 10],
                [1, 2, 1],
                [3, np.nan, 0],
                [8, 7, 9]
            ]
        ],
        dims=['c', 'a', 'b'],
        coords=coords
    ).chunk((3, 1))
    result = Algorithms.multi_rank(
        new_data=arr,
        dim=dim,
        tie_dim="c"
    )

    data = [
        [np.nan, 3., 3.],
        [1., 4., 1.],
        [2., 2., 2.],
        [np.nan, 1., 4.],
        [3., np.nan, 5.]
    ]
    if dim == "b":
        data = [
            [np.nan, 2., 1.],
            [2., 3., 1.],
            [3., 2., 1.],
            [np.nan, 1., 2.],
            [1., np.nan, 2.]
        ]
    expected = xr.DataArray(
        data,
        coords={k: v for k, v in coords.items() if k != "c"},
        dims=["a", "b"]
    )
    assert result.isel(c=0, drop=True).equals(expected)


def test_rolling_along_axis():
    arr = xr.DataArray(
        [
            [1, np.nan, 3],
            [np.nan, 4, 6],
            [np.nan, 5, np.nan],
            [3, np.nan, 7],
            [7, 6, np.nan]
        ],
        dims=['a', 'b'],
        coords={'a': list(range(5)), 'b': list(range(3))}
    ).chunk((3, 1))
    df = pd.DataFrame(arr.values.T, arr.b.values, arr.a.values).stack(dropna=False)
    for window in range(1, 4):
        for min_periods in [None] + list(range(1, window)):
            for drop_nan in [True, False]:
                for fill_method in [None, 'ffill']:
                    rolling_arr = Algorithms.rolling_along_axis(
                        arr,
                        window=window,
                        dim='a',
                        operator='mean',
                        min_periods=min_periods,
                        drop_nan=drop_nan,
                        fill_method=fill_method
                    )

                    expected = df
                    if drop_nan:
                        expected = expected.dropna()
                    expected = expected.groupby(level=0).rolling(window=window, min_periods=min_periods).mean()
                    expected = expected.droplevel(0).unstack(0)

                    if fill_method == 'ffill' and drop_nan:
                        expected.ffill(inplace=True)

                    expected = xr.DataArray(expected.values, coords=arr.coords, dims=arr.dims)
                    assert expected.equals(rolling_arr)


@pytest.mark.parametrize(
    'default_replace',
    [np.nan, None, 5.3]
)
def test_replace(default_replace):
    arr = xr.DataArray(
        [
            [1, 2, 3],
            [4, 4, 1],
            [5, 2, 3],
            [np.nan, 3, 0],
            [8, 7, 9]
        ],
        dims=['a', 'b'],
        coords={'a': list(range(5)), 'b': list(range(3))}
    ).chunk((3, 1))

    df = pd.DataFrame(arr.values, index=arr.a.values, columns=arr.b.values)

    to_replace = {
        1: 11,
        2: 12,
        3: 13,
        4: 14,
        5: 15,
        7: 16
    }

    new_data = Algorithms.replace(
        new_data=arr,
        to_replace=to_replace,
        dtype=float,
        default_replace=default_replace
    )

    if default_replace is not None:
        df[~df.isin(list(to_replace.keys()))] = default_replace

    replaced_df = df.replace(to_replace)

    assert xr.DataArray(
        replaced_df.values,
        coords={'a': replaced_df.index, 'b': replaced_df.columns},
        dims=['a', 'b']
    ).equals(
        new_data
    )


def test_vindex():
    arr = xr.DataArray(
        [
            [[1, 3], [4, 1], [5, 2]],
            [[4, 2], [5, 1], [3, 4]],
            [[5, 1], [2, -1], [3, -5]],
            [[np.nan, 1], [3, 3], [0, -2]],
            [[8, 3], [7, 5], [9, 11]]
        ],
        dims=['a', 'b', 'c'],
        coords={'a': list(range(5)), 'b': list(range(3)), 'c': list(range(2))}
    ).chunk((3, 2, 1))

    for i_coord in [None, [0, 3, 1, 4], [3, 4, 2, 1], [1, 1, 1]]:
        for j_coord in [None, [1, 0, 2], [2, 1, 0], [0, 0, 0], [1, 0]]:
            for k_coord in [None, [0, 1], [1, 0], [0], [1]]:
                coords = {'a': i_coord, 'b': j_coord, 'c': k_coord}
                coords = {k: v for k, v in coords.items() if v is not None}
                if len(coords) == 0:
                    continue
                expected = arr.reindex(coords)
                result = Algorithms.vindex(arr, coords)
                assert expected.equals(result)


@pytest.mark.parametrize(
    'dim, keep_shape, func',
    [
        ('a', False, "max"),
        ('b', False, "max"),
        ('a', True, "max"),
        ('b', True, "max"),
    ]
)
def test_apply_on_groups(dim, keep_shape, func):
    arr = xr.DataArray(
        [
            [1, 2, 3, 4, 3],
            [4, 4, 1, 3, 5],
            [5, 2, 3, 2, 1],
            [np.nan, 3, 0, 5, 4],
            [8, 7, 9, 6, 7]
        ],
        dims=['a', 'b'],
        coords={'a': [1, 2, 3, 4, 5], 'b': [0, 1, 2, 3, 4]}
    ).chunk((3, 2))
    grouper = {
        'a': [1, 5, 5, 0, 1],
        'b': [0, 1, 1, 0, -1]
    }
    groups = {k: v for k, v in zip(arr.coords[dim].values, grouper[dim])}

    result = Algorithms.apply_on_groups(
        arr, groups=groups, dim=dim, func=func, keep_shape=keep_shape
    )

    expected = arr.to_pandas()
    axis = 0 if dim == "a" else 1

    if keep_shape:
        expected = expected.groupby(groups, axis=axis).transform(func)
    else:
        expected = getattr(expected.groupby(groups, axis=axis), func)()

    expected = xr.DataArray(
        expected.values,
        coords=result.coords,
        dims=result.dims
    )
    assert expected.equals(result)


@pytest.mark.parametrize(
    'dim, keep_shape, func',
    [
        ('a', True, 'rank'),
        ('a', False, 'max'),
        ('b', False, 'max'),
        ('a', True, 'max'),
        ('b', True, 'max'),
    ]
)
def test_apply_on_groups_array(dim, keep_shape, func):
    arr = xr.DataArray(
        [
            [1, 2, 3, 4, 3],
            [4, 4, 1, 3, 5],
            [5, 2, 3, 2, 1],
            [np.nan, 3, 7, 5, 4],
            [8, 7, 9, 6, 7]
        ],
        dims=['a', 'b'],
        coords={'a': [1, 2, 3, 4, 5], 'b': [0, 1, 2, 3, 4]}
    ).chunk((3, 2))
    groups = xr.DataArray(
        [
            [0, 0, 2, 1, 0],
            [10, 5, 1, 9, 2],
            [4, 4, 2, 100, 2],
            [0, 3, 2, 2, 3],
            [8, 7, 7, 7, 7]
        ],
        dims=['a', 'b'],
        coords={'a': [1, 2, 3, 4, 5], 'b': [0, 1, 2, 3, 4]}
    ).chunk((3, 2))
    unique_groups = np.unique(groups.values)
    axis = arr.dims.index(dim)

    result = Algorithms.apply_on_groups(
        arr, groups=groups, dim=dim, func=func, keep_shape=keep_shape
    )

    iterate_dim = 'b' if dim == 'a' else 'a'
    for coord in arr.coords[iterate_dim].values:
        x = arr.sel({iterate_dim: coord}, drop=True)
        grouper = groups.sel({iterate_dim: coord}, drop=True).compute()
        r = result.sel({iterate_dim: coord}, drop=True)
        kwargs = {"method": "first", "axis": axis} if func == "rank" else {}
        s = x.to_pandas()
        expected = getattr(s.groupby(grouper.to_pandas()), func)(**kwargs).to_xarray()
        if dim not in expected.dims:
            expected = expected.rename({expected.dims[0]: dim})
        if keep_shape:
            if func != "rank":
                expected = expected.reindex({dim: grouper.values})
                expected.coords[dim] = arr.coords[dim].values
        else:
            assert np.all(r.coords[dim].values == unique_groups)
            assert r.sum() == expected.sum()
            r = r.sel({dim: expected.coords[dim]})
        assert expected.equals(r)


@pytest.mark.parametrize('dim', ['a', 'b'])
def test_merge_duplicates_coord(dim):
    arr = xr.DataArray(
        [
            [1, 2, 3, 4, 3],
            [4, 4, 1, 3, 5],
            [5, 2, 3, 2, 1],
            [np.nan, 3, 0, 5, 4],
            [8, 7, 9, 6, 7]
        ],
        dims=['a', 'b'],
        coords={'a': [1, 5, 5, 0, 1], 'b': [0, 1, 1, 0, -1]}
    ).chunk((3, 2))

    g = arr.groupby(dim).max(dim)
    arr = Algorithms.merge_duplicates_coord(arr, dim, 'max')
    assert g.equals(arr)


@pytest.mark.parametrize(
    'dim, ascending, func',
    [
        ('a', True, 'cumsum'),
        ('a', False, 'cumsum'),
        ('b', True, 'cumsum'),
        ('b', False, 'cumsum'),
        ('b', False, 'cumprod'),
    ]
)
def test_cumulative_on_sort(dim, ascending, func):
    arr = xr.DataArray(
        [
            [1, 2, 3],
            [4, 4, 1],
            [5, 2, 3],
            [np.nan, 3, 0],
            [8, 7, 9]
        ],
        dims=['a', 'b'],
        coords={'a': list(range(5)), 'b': list(range(3))}
    ).chunk((5, 3))
    result = Algorithms.cumulative_on_sort(
        arr,
        dim=dim,
        func=getattr(np, f"nan{func}"),
        ascending=ascending
    )

    expected = arr.to_series()
    expected = expected.groupby(
        level="a" if dim == "b" else "b",
        group_keys=False
    ).apply(
        lambda x: getattr(x.sort_values(ascending=ascending), func)()
    )
    expected = expected.unstack()
    expected = xr.DataArray(
        expected.values,
        dims=result.dims,
        coords=result.coords
    )

    assert result.equals(expected)


if __name__ == "__main__":
    test = TestAlgorithms()
    # test.test_ffill()
    test.test_replace()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
