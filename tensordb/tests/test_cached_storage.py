import fsspec
import numpy as np
import pytest
import xarray as xr

from tensordb.storages import ZarrStorage
from tensordb.storages.cached_storage import CachedStorage


# TODO: Add more tests for the update cases

class TestCachedTensor:
    @pytest.fixture(autouse=True)
    def setup_tests(self, tmpdir):
        sub_path = tmpdir.strpath
        storage = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='zarr_cache',
            dataset_names='cached_test',
            chunks={'index': 3, 'columns': 2},
        )
        self.cached_storage = CachedStorage(
            storage=storage,
            max_cached_in_dim=3,
            dim='index'
        )

        self.arr = xr.DataArray(
            data=np.array([
                [1, 2, 7, 4, 5],
                [np.nan, 3, 5, 5, 6],
                [3, 3, np.nan, 5, 6],
                [np.nan, 3, 10, 5, 6],
                [np.nan, 7, 8, 5, 6],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [0, 1, 2, 3, 4], 'columns': [0, 1, 2, 3, 4]},
        )

    def test_append(self):
        self.cached_storage.append(self.arr.isel(index=[0]))
        self.cached_storage.append(self.arr.isel(index=[1]))
        self.cached_storage.append(self.arr.isel(index=[2]))

        assert self.cached_storage._cached_count == 3
        assert len(self.cached_storage._cache['append']['new_data']) == 3

        self.cached_storage.append(self.arr.isel(index=[3]))
        assert self.cached_storage._cached_count == 0
        assert len(self.cached_storage._cache['append']['new_data']) == 0

        self.cached_storage.append(self.arr.isel(index=[4]))
        self.cached_storage.close()
        assert self.cached_storage._cached_count == 0
        assert len(self.cached_storage._cache['append']['new_data']) == 0

        assert self.cached_storage.read().equals(self.arr)

    def test_store(self):
        self.cached_storage.store(self.arr.isel(index=[0]))
        self.cached_storage.append(self.arr.isel(index=[1]))
        self.cached_storage.append(self.arr.isel(index=[2]))
        assert self.cached_storage._cached_count == 3
        assert len(self.cached_storage._cache['store']['new_data']) == 3

        self.cached_storage.store(self.arr.isel(index=[3, 4]))
        assert self.cached_storage._cached_count == 2
        assert len(self.cached_storage._cache['store']['new_data']) == 1
        self.cached_storage.close()
        assert self.cached_storage._cached_count == 0
        assert len(self.cached_storage._cache['store']['new_data']) == 0

        assert self.cached_storage.read().equals(self.arr.isel(index=[3, 4]))


if __name__ == "__main__":
    test = TestCachedTensor()
    # test.test_append()
    test.test_store()
