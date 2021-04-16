import xarray
import numpy as np
import os
import shutil

from tensordb.file_handlers import ZarrStorage
from tensordb.core.utils import compare_dataset
from tensordb.config.config_root_dir import TEST_DIR_ZARR


# TODO: Improve the tests and add the backup test


def get_default_zarr_storage():
    return ZarrStorage(
        base_path=TEST_DIR_ZARR,
        path='first_test',
        name='data_test',
        chunks={'index': 3, 'columns': 2},
        dims=['index', 'columns'],
    )


class TestZarrStore:
    arr = xarray.DataArray(
        data=np.array([
            [1, 2, 7, 4, 5],
            [2, 3, 5, 5, 6],
            [3, 3, 11, 5, 6],
            [4, 3, 10, 5, 6],
            [5, 7, 8, 5, 6],
        ], dtype=float),
        dims=['index', 'columns'],
        coords={'index': [0, 1, 2, 3, 4], 'columns': [0, 1, 2, 3, 4]},
    )

    arr2 = xarray.DataArray(
        data=np.array([
            [1, 2, 7, 4, 5, 10, 13],
            [2, 3, 5, 5, 6, 11, 15],
            [2, 3, 5, 5, 6, 11, 15],
        ], dtype=float),
        dims=['index', 'columns'],
        coords={'index': [6, 7, 8], 'columns': [0, 1, 2, 3, 4, 5, 6]},
    )

    def test_store_data(self):
        a = get_default_zarr_storage()
        a.store(TestZarrStore.arr)
        dataset = a.read()
        assert compare_dataset(dataset, TestZarrStore.arr)

    def test_append_data(self):
        a = get_default_zarr_storage()
        a.s3_handler = None
        if os.path.exists(a.local_path):
            shutil.rmtree(a.local_path)

        arr = TestZarrStore.arr.to_dataset(name='data_test')
        for i in range(5):
            a.append(arr.isel(index=[i]))

        arr2 = TestZarrStore.arr2.to_dataset(name='data_test')
        for i in range(3):
            a.append(arr2.isel(index=[i]))

        total_data = xarray.concat([arr, arr2], dim='index')
        dataset = a.read()
        assert compare_dataset(dataset, total_data)

    def test_update_data(self):
        self.test_store_data()
        a = get_default_zarr_storage()
        a.update(TestZarrStore.arr + 5)
        dataset = a.read()
        assert compare_dataset(dataset, TestZarrStore.arr + 5)

    # def test_backup(self):
    #     """
    #     TODO: Improve this test
    #     """
    #     a = get_default_zarr_storage()
    #     a.store(TestZarrStore.arr)
    #     a.backup()
    #     shutil.rmtree(a.local_path)
    #     a.update_from_backup()
    #     dataset = a.read()
    #     assert compare_dataset(dataset, TestZarrStore.arr)


if __name__ == "__main__":
    test = TestZarrStore()
    test.test_store_data()
    # test.test_append_data()
    # test.test_update_data()
    # test.test_backup()
