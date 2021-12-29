import pandas as pd
import xarray as xr
import os
import numpy as np
import pytest

from typing import List, Dict
from loguru import logger

from tensordb.algorithms import Algorithms


# TODO: Add more tests for the dataset cases


class TestAlgorithms:

    def test_ffill(self):
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

    def test_rolling_along_axis(self):
        arr = xr.DataArray(
            [
                [1,         np.nan, 3],
                [np.nan,    4,      6],
                [np.nan,    5,      np.nan],
                [3,         np.nan, 7],
                [7,         6,      np.nan]
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


if __name__ == "__main__":
    test = TestAlgorithms()
    test.test_ffill()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
