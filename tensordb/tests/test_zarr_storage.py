import xarray as xr
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
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='zarr',
            data_names='data_test',
            chunks={'index': 3, 'columns': 2},
        )
        self.storage_dataset = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='zarr_dataset',
            data_names=['a', 'b', 'c'],
            chunks={'index': 3, 'columns': 2}
        )
        self.storage_sorted_unique = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='zarr',
            data_names='data_test',
            chunks={'index': 3, 'columns': 2},
            unique_coords=True,
            sorted_coords={'index': False, 'columns': False}
        )
        self.storage_dataset_sorted_unique = ZarrStorage(
            base_map=fsspec.get_mapper(sub_path),
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='zarr_dataset',
            data_names=['a', 'b', 'c'],
            chunks={'index': 3, 'columns': 2},
            unique_coords=True,
            sorted_coords={'index': False, 'columns': False}
        )
        self.arr = xr.DataArray(
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

        self.arr2 = xr.DataArray(
            data=np.array([
                [1, 2, 7, 4, 5, 10, 13],
                [2, 3, 5, 5, 6, 11, 15],
                [2, 3, 5, 5, 6, 11, 15],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [6, 7, 8], 'columns': [0, 1, 2, 3, 4, 5, 6]},
        )

        self.arr3 = xr.DataArray(
            data=np.array([
                [1, 2, 3, 4, 5],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [5], 'columns': [0, 1, 2, 3, 4]},
        )

        self.arr4 = self.arr.astype(float) + 5
        self.arr5 = self.arr.astype(np.uint) + 3

        self.dataset = xr.Dataset(
            data_vars=dict(
                a=self.arr,
                b=self.arr4,
                c=self.arr5
            )
        )

    @pytest.mark.parametrize('keep_order', [True, False])
    def test_store_data(self, keep_order: bool):
        storage = self.storage_sorted_unique if keep_order else self.storage
        storage.store(self.arr2)
        if keep_order:
            assert storage.read().equals(
                self.arr2.sel(index=self.arr2.index[::-1], columns=self.arr2.columns[::-1])
            )
        else:
            assert storage.read().equals(self.arr2)

    @pytest.mark.parametrize('keep_order', [True, False])
    @pytest.mark.parametrize('as_dask', [False, False])
    def test_append_data(self, keep_order: bool, as_dask: bool):
        storage = self.storage_sorted_unique if keep_order else self.storage

        storage.delete_tensor()
        arr, arr2 = self.arr, self.arr2
        if as_dask:
            arr, arr2 = arr.chunk((2, 3)), arr2.chunk((1, 4))

        for i in range(len(arr.index)):
            storage.append(arr.isel(index=[i]))

        for i in range(len(arr2.index)):
            storage.append(arr2.isel(index=[i]))

        total_data = xr.concat([arr, arr2], dim='index')
        if keep_order:
            assert storage.read().equals(
                total_data.sel(index=total_data.index[::-1], columns=total_data.columns[::-1])
            )
        else:
            assert storage.read().equals(total_data)

    def test_update_data(self):
        self.storage.store(self.arr)

        expected = xr.concat([
            self.arr.sel(index=slice(0, 1)),
            self.arr.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage.update(expected.sel(index=slice(2, None)))

        assert self.storage.read().equals(expected)

    @pytest.mark.parametrize('keep_order', [True, False])
    def test_store_dataset(self, keep_order: bool):
        storage_dataset = self.storage_dataset_sorted_unique if keep_order else self.storage_dataset
        storage_dataset.store(self.dataset)

        if keep_order:
            assert storage_dataset.read().equals(
                self.dataset.sel(index=self.dataset.index[::-1], columns=self.dataset.columns[::-1])
            )
        else:
            assert storage_dataset.read().equals(self.dataset)

    @pytest.mark.parametrize('keep_order', [True, False])
    def test_append_dataset(self, keep_order: bool):
        storage_dataset = self.storage_dataset_sorted_unique if keep_order else self.storage_dataset
        storage_dataset.store(self.dataset)
        dataset = self.dataset.reindex(
            index=list(self.dataset.index.values) + [self.dataset.index.values[-1] + 1],
            fill_value=1
        )
        storage_dataset.append(dataset.isel(index=[-1]))
        if keep_order:
            assert storage_dataset.read().equals(
                dataset.sel(index=dataset.index[::-1], columns=dataset.columns[::-1])
            )
        else:
            assert storage_dataset.read().equals(dataset)

    def test_update_dataset(self):
        self.storage_dataset.store(self.dataset)
        expected = xr.concat([
            self.dataset.sel(index=slice(0, 1)),
            self.dataset.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage_dataset.update(expected.sel(index=slice(2, None)))
        assert self.storage_dataset.read().equals(expected)

    def test_drop_data(self):
        self.storage.store(self.arr)
        coords = {'index': [0, 2, 4], 'columns': [1, 3]}
        self.storage.drop(coords)
        assert self.storage.read().equals(
            self.arr.drop_sel(coords)
        )


if __name__ == "__main__":
    test = TestZarrStorage()
    # test.test_store_data()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
