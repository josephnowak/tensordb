import xarray as xr
import os
import numpy as np
import pytest

from typing import List, Dict
from loguru import logger

from tensordb.algorithms import ffill, rank


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
        assert ffill(arr, limit=2, dim='b').equals(arr.compute().ffill('b', limit=2))
        assert ffill(arr, limit=2, dim='b', until_last_valid=True).equals(
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


if __name__ == "__main__":
    test = TestAlgorithms()
    test.test_ffill()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
