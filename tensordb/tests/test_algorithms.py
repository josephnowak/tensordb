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
    'method, rank_nan',
    [
        ('average', True),
        ('min', False),
        ('max', False),
        ('ordinal', True),
        ('dense', False)
    ]
)
def test_rank(method, rank_nan):
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
        rank_nan=rank_nan
    )
    rank_pandas_method = 'first' if method == 'ordinal' else method
    expected = df.rank(axis=1, method=rank_pandas_method, na_option='bottom' if rank_nan else 'keep')
    assert result.equals(
        xr.DataArray(
            expected.values,
            dims=arr.dims,
            coords=arr.coords
        )
    )


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


def test_replace():
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

    for method in ('vectorized', 'unique'):
        new_data = Algorithms.replace(
            new_data=arr,
            # method=method,
            to_replace=to_replace,
            dtype=float,
        )
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
    'dim, keep_shape, output_dim',
    [
        ('a', False, None),
        ('b', False, None),
        ('a', True, None),
        ('b', True, None),
        ('b', True, 'h')
    ]
)
def test_apply_on_groups(dim, keep_shape, output_dim):
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

    g = arr.groupby(xr.IndexVariable(dim, grouper[dim])).max(dim)
    arr = Algorithms.apply_on_groups(
        arr, groups=groups, dim=dim, func='nanmax', keep_shape=keep_shape, output_dim=output_dim
    )
    if output_dim:
        g = g.rename({dim: output_dim})
        grouper[output_dim] = grouper[dim]
        dim = output_dim

    if keep_shape:
        g = g.reindex({dim: grouper[dim]})
        g.coords[dim] = arr.coords[dim].values

    assert g.equals(arr)


@pytest.mark.parametrize(
    'dim, keep_shape, output_dim',
    [
        ('a', False, None),
        ('b', False, None),
        ('a', True, None),
        ('b', True, None),
        ('b', True, 'h')
    ]
)
def test_apply_on_groups_array(dim, keep_shape, output_dim):
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

    result = Algorithms.apply_on_groups(
        arr, groups=groups, dim=dim, func='nanmax', keep_shape=keep_shape, output_dim=output_dim
    )

    iterate_dim = 'b' if dim == 'a' else 'a'
    for i in range(arr.sizes[iterate_dim]):
        x = arr.isel({iterate_dim: i})
        grouper = groups.isel({iterate_dim: i})
        r = result.isel({iterate_dim: i})
        g = x.groupby(xr.IndexVariable(dim, grouper)).max(dim)
        if output_dim:
            g = g.rename({dim: output_dim})
        else:
            output_dim = dim

        if keep_shape:
            g = g.reindex({output_dim: grouper.values})
            g.coords[output_dim] = arr.coords[dim].values

        if not keep_shape:
            assert np.all(r.coords[dim].values == unique_groups)
            assert r.sum() == g.sum()
            r = r.sel({dim: g.coords[dim]})
        assert g.equals(r)


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
    arr = Algorithms.merge_duplicates_coord(arr, dim, 'nanmax')
    assert g.equals(arr)


if __name__ == "__main__":
    test = TestAlgorithms()
    # test.test_ffill()
    test.test_replace()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
