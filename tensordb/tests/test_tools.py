import dask
import numpy as np
import pytest
import xarray as xr

from tensordb.utils.tools import groupby_chunks, xarray_from_func


class TestTools:
    @pytest.fixture(autouse=True)
    def setup_tests(self):
        self.data_array = xr.DataArray(
            np.arange(56).reshape((7, 8)).astype(np.float64),
            dims=["a", "b"],
            coords={"a": list(range(7)), "b": list(range(8))},
        )
        self.dataset = xr.Dataset(
            {"first": self.data_array, "second": self.data_array + 10}
        )
        # TODO: Once this issue is fixed https://github.com/pydata/xarray/issues/7059 this lock should be dropped
        self.lock = dask.utils.SerializableLock()

    def read_by_coords(self, arr: xr.DataArray) -> xr.DataArray:
        with self.lock:
            return self.data_array.sel(arr.coords)

    def read_by_coords_dataset(self, arr: xr.Dataset) -> xr.Dataset:
        with self.lock:
            return self.dataset.loc[arr.coords]

    def test_xarray_from_func_data_array(self):
        data = xarray_from_func(
            self.read_by_coords,
            dims=["a", "b"],
            coords={"a": list(range(6)), "b": list(range(8))},
            chunks=[2, 3],
            dtypes=np.float64,
            func_parameters={},
        )
        assert data.equals(self.data_array.sel(**data.coords))

    def test_xarray_from_func_dataset(self):
        data = xarray_from_func(
            self.read_by_coords_dataset,
            dims=["a", "b"],
            coords={"a": list(range(6)), "b": list(range(8))},
            chunks=[2, 3],
            dtypes=[np.float64, np.float64],
            data_names=["first", "second"],
            func_parameters={},
        )
        assert data.equals(self.dataset.sel(data.coords))

    def test_groupby_chunks(self):
        e = {"a": 0, "b": 1, "c": 0, "d": 0, "e": 0, "m": 1, "g": 2, "l": 2}
        result = list(
            groupby_chunks(list(e), {0: 2, 1: 1}, lambda x: e[x], lambda x: (e[x], x))
        )
        assert result == [["a", "c", "b", "g", "l"], ["d", "e", "m"]]

        result = list(
            groupby_chunks(
                list(e), {0: 2, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)
            )
        )
        assert result == [["a", "c", "b", "m", "g"], ["d", "e", "l"]]

        e["f"] = 1
        result = list(
            groupby_chunks(
                list(e), {0: 2, 1: 1, 2: 2}, lambda x: e[x], lambda x: (e[x], x)
            )
        )
        assert result == [["a", "c", "b", "g", "l"], ["d", "e", "f"], ["m"]]

        result = list(
            groupby_chunks(
                list(e), {0: 1, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)
            )
        )
        assert result == [["a", "b", "f", "g"], ["c", "m", "l"], ["d"], ["e"]]

        result = list(
            groupby_chunks(
                list(e), {0: 3, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)
            )
        )
        assert result == [["a", "c", "d", "b", "f", "g"], ["e", "m", "l"]]


if __name__ == "__main__":
    test = TestTools()
    test.test_groupby_chunks()
