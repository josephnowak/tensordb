import fsspec
import xarray
import numpy as np

from loguru import logger

from tensordb.core import CachedTensorHandler
from tensordb.file_handlers import ZarrStorage
from tensordb.config.config_root_dir import TEST_DIR_CACHED_TENSOR


def get_default_zarr_storage():
    return ZarrStorage(
        local_base_map=fsspec.get_mapper(TEST_DIR_CACHED_TENSOR),
        backup_base_map=fsspec.get_mapper(TEST_DIR_CACHED_TENSOR + '/backup'),
        path='cached_test',
        name='data',
        chunks={'index': 3, 'columns': 2},
    )


def get_cached_tensor():
    return CachedTensorHandler(
        file_handler=get_default_zarr_storage(),
        max_cached_in_dim=3,
        dim='index'
    )


class TestCachedTensor:
    arr = xarray.DataArray(
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
        cached_tensor = get_cached_tensor()
        arr = TestCachedTensor.arr
        cached_tensor.append(arr.isel(index=[0]))
        cached_tensor.append(arr.isel(index=[1]))
        cached_tensor.append(arr.isel(index=[2]))

        assert cached_tensor._cached_count == 3
        assert len(cached_tensor._cached_operations['append']['new_data']) == 3

        cached_tensor.append(arr.isel(index=[3]))
        assert cached_tensor._cached_count == 0
        assert len(cached_tensor._cached_operations['append']['new_data']) == 0

        cached_tensor.append(arr.isel(index=[4]))
        cached_tensor.close()
        assert cached_tensor._cached_count == 0
        assert len(cached_tensor._cached_operations['append']['new_data']) == 0

        assert cached_tensor.read().equals(arr)
        cached_tensor.delete_file(True)

    def test_store(self):
        cached_tensor = get_cached_tensor()
        arr = TestCachedTensor.arr
        cached_tensor.store(arr.isel(index=[0]))
        cached_tensor.append(arr.isel(index=[1]))
        cached_tensor.append(arr.isel(index=[2]))
        assert cached_tensor._cached_count == 3
        assert len(cached_tensor._cached_operations['store']['new_data']) == 3

        cached_tensor.store(arr.isel(index=[3, 4]))
        assert cached_tensor._cached_count == 2
        assert len(cached_tensor._cached_operations['store']['new_data']) == 1
        cached_tensor.close()
        assert cached_tensor._cached_count == 0
        assert len(cached_tensor._cached_operations['store']['new_data']) == 0

        assert cached_tensor.read().equals(arr.isel(index=[3, 4]))
        cached_tensor.delete_file(True)


if __name__ == "__main__":
    test = TestCachedTensor()
    # test.test_append()
    test.test_store()



