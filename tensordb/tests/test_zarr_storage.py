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
            local_base_map=fsspec.get_mapper(sub_path),
            backup_base_map=fsspec.get_mapper(sub_path + '/backup'),
            path='zarr',
            data_names='data_test',
            chunks={'index': 3, 'columns': 2},
        )
        self.storage_dataset = ZarrStorage(
            local_base_map=fsspec.get_mapper(sub_path),
            backup_base_map=fsspec.get_mapper(sub_path + '/backup'),
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
        self.storage.store(self.arr2, remote=True)
        self.storage.store(self.arr, remote=False)

        # check that the backup does not affect the local tensor
        assert self.storage.read(remote=True).equals(self.arr2)
        assert self.storage.read(remote=False).equals(self.arr)

    def test_read_autoupdate(self):
        self.storage.store(self.arr, remote=True)
        assert self.storage.read(remote=False).equals(self.arr)

        self.storage.append(self.arr3, remote=True)
        assert self.storage.read(remote=False).equals(xarray.concat([self.arr, self.arr3], dim='index'))

    @pytest.mark.parametrize("remote", (True, False))
    def test_append_data(self, remote):
        total_data = xarray.concat([self.arr, self.arr2], dim='index')

        for i in range(5):
            self.storage.append(self.arr.isel(index=[i]), remote=remote)
        for i in range(3):
            self.storage.append(self.arr2.isel(index=[i]), remote=remote)

        assert self.storage.read(remote=remote).equals(total_data)
        self.storage.delete_tensor(only_local=False)

    @pytest.mark.parametrize("remote", (True, False))
    def test_update_data(self, remote):
        self.storage.store(self.arr, remote=remote)

        expected = xarray.concat([
            self.arr.sel(index=slice(0, 1)),
            self.arr.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage.update(expected.sel(index=slice(2, None)), remote=remote)

        assert self.storage.read(remote=remote).equals(expected)
        self.storage.delete_tensor(only_local=False)

    def test_backup(self):
        self.storage.store(self.arr, remote=False)
        self.storage.backup()
        assert self.storage.read(remote=True).equals(self.arr)
        assert self.storage.read(remote=True).equals(self.storage.read(remote=False))

        self.storage.append(self.arr3, remote=False)
        assert self.storage.backup()
        assert self.storage.read(remote=True).equals(xarray.concat([self.arr, self.arr3], dim='index'))
        assert self.storage.read(remote=True).equals(self.storage.read(remote=False))

        #
        self.storage.store(self.arr, remote=True)
        self.storage.append(self.arr3, remote=False)
        assert self.storage.read(remote=False).equals(xarray.concat([self.arr, self.arr3], dim='index'))

        # the backup of the local copy must fail if the original backup was modified first
        self.storage.store(self.arr, remote=True)
        self.storage.update_from_backup(force_update=True)
        self.storage.append(self.arr3, remote=False)
        self.storage.append(self.arr3, remote=True)
        try:
            self.storage.backup()
            assert False
        except ValueError:
            assert True

    @pytest.mark.parametrize("remote", (True, False))
    def test_store_dataset(self, remote):
        self.storage_dataset.store(self.dataset, remote=remote)
        assert self.storage_dataset.read(remote=remote).equals(self.dataset)

    @pytest.mark.parametrize("remote", (True, False))
    def test_append_dataset(self, remote):
        self.storage_dataset.store(self.dataset, remote=remote)
        dataset = self.dataset.reindex(
            index=list(self.dataset.index.values) + [self.dataset.index.values[-1] + 1],
            fill_value=1
        )
        self.storage_dataset.append(dataset.isel(index=[-1]), remote=remote)
        assert self.storage_dataset.read(remote=remote).equals(dataset)

    @pytest.mark.parametrize("remote", (True, False))
    def test_update_dataset(self, remote):
        self.storage_dataset.store(self.dataset, remote=remote)
        expected = xarray.concat([
            self.dataset.sel(index=slice(0, 1)),
            self.dataset.sel(index=slice(2, None)) + 5
        ], dim='index')
        self.storage_dataset.update(expected.sel(index=slice(2, None)), remote=remote)
        assert self.storage_dataset.read(remote=remote).equals(expected)

    # @pytest.fixture(scope="session", autouse=True)
    # def cleanup(self, request):
    #     """Cleanup a testing directory once we are finished."""
    #
    #     def remove_test_dir():
    #         shutil.rmtree(test_path)
    #
    #     request.addfinalizer(remove_test_dir)


if __name__ == "__main__":
    test = TestZarrStorage()
    # test.test_store_data()
    test.test_read_autoupdate()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
    # test.test_different_storages()
