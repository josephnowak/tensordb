import xarray
import os
import numpy as np
import pytest
import zarr
import fsspec
import shutil

from loguru import logger

from tensordb.storages import ZarrStorage


# TODO: Add more tests for the dataset cases


class TestZarrStorage:

    @pytest.fixture(autouse=True)
    def setup_tests(self, tmpdir):
        sub_path = tmpdir.strpath
        self.storage = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            path='zarr',
            data_names='data_test',
            chunks={'index': 3, 'columns': 2},
        )
        self.storage_dataset = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            path='zarr_dataset',
            data_names=['a', 'b', 'c'],
            chunks={'index': 3, 'columns': 2},
        )
        self.arr = xarray.DataArray(
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

        self.arr2 = xarray.DataArray(
            data=np.array([
                [1, 2, 7, 4, 5, 10, 13],
                [2, 3, 5, 5, 6, 11, 15],
                [2, 3, 5, 5, 6, 11, 15],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [6, 7, 8], 'columns': [0, 1, 2, 3, 4, 5, 6]},
        )

        self.arr3 = xarray.DataArray(
            data=np.array([
                [1, 2, 3, 4, 5],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [5], 'columns': [0, 1, 2, 3, 4]},
        )

        self.arr4 = self.arr.astype(float) + 5
        self.arr5 = self.arr.astype(np.uint) + 3

        self.dataset = xarray.Dataset(
            data_vars=dict(
                a=self.arr,
                b=self.arr4,
                c=self.arr5
            )
        )

    def test_store_data(self):
        self.storage.store(self.arr2)
        assert self.storage.read().equals(self.arr2)

    def test_append_data(self):
        self.storage.delete_tensor()

        total_data = xarray.concat([self.arr, self.arr2], dim='index')

        for i in range(len(self.arr.index)):
            self.storage.append(self.arr.isel(index=[i]))
        for i in range(len(self.arr2.index)):
            self.storage.append(self.arr2.isel(index=[i]))

        assert self.storage.read().equals(total_data)

    def test_update_data(self):
        self.storage.store(self.arr)

        expected = xarray.concat([
            self.arr.sel(index=slice(0, 1)),
            self.arr.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage.update(expected.sel(index=slice(2, None)))

        assert self.storage.read().equals(expected)

    def test_store_dataset(self):
        self.storage_dataset.store(self.dataset)
        assert self.storage_dataset.read().equals(self.dataset)

    def test_append_dataset(self):
        self.storage_dataset.store(self.dataset)
        dataset = self.dataset.reindex(
            index=list(self.dataset.index.values) + [self.dataset.index.values[-1] + 1],
            fill_value=1
        )
        self.storage_dataset.append(dataset.isel(index=[-1]))
        assert self.storage_dataset.read().equals(dataset)

    def test_update_dataset(self):
        self.storage_dataset.store(self.dataset)
        expected = xarray.concat([
            self.dataset.sel(index=slice(0, 1)),
            self.dataset.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage_dataset.update(expected.sel(index=slice(2, None)))
        assert self.storage_dataset.read().equals(expected)


if __name__ == "__main__":
    test = TestZarrStorage()
    # test.test_store_data()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
