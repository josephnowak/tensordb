import fsspec
import xarray
import numpy as np
import pytest
import os

from loguru import logger

from tensordb import TensorClient

# TODO: Add more tests that validate the internal behaviour of the storage settings

tensors_definition = {
    'data_four': {
        'read': {
            'substitute_method': 'read_from_formula',
        },
        'read_from_formula': {
            'formula': "new_data = (`data_one` * `data_two`).rolling({'index': 3}).sum()",
            'use_exec': True
        }
    },

    'data_ffill': {
        'store': {
            'data_transformation': ['read_from_formula', 'ffill'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'ffill': {
            'dim': 'index'
        }
    },
    'last_valid_dim': {
        'store': {
            'data_transformation': ['read_from_formula', 'last_valid_dim'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'last_valid_dim': {
            'dim': "index",
        }
    },
    'data_replace_by_last_valid': {
        'store': {
            'data_transformation': [
                'read_from_formula',
                ['read_from_formula', {
                    'formula': '`data_one`.notnull().cumsum(dim="index").idxmax(dim="index")',
                    'result_name': 'last_valid_data'
                }],
                'ffill',
                'replace_by_last_valid'
            ],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'ffill': {
            'dim': 'index'
        },
        'replace_by_last_valid': {
            'dim': 'index',
        }
    },

    'reindex_data': {
        'store': {
            'data_transformation': ['read_from_formula', 'reindex'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'reindex': {
            'dims_to_reindex': ["index"],
            'reindex_data': 'data_three',
            'method_fill_value': 'ffill'
        }
    },
    'overwrite_append_data': {
        'store': {
            'data_transformation': ['read_from_formula'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'append': {
            'substitute_method': 'store'
        }
    },
    'specific_definition': {
        'store': {
            'data_transformation': [
                ['read_from_formula', {'formula': "`data_one`"}],
                ['read_from_formula', {'formula': "new_data * `data_one`"}],
            ],
        },
    },
    'different_client': {
        'storage': {
            'storage_name': 'json_storage'
        }
    }
}


def create_dummy_array(n_rows, n_cols, coords=None, dtype=None) -> xarray.DataArray:
    coords = coords
    if coords is None:
        dtype = dtype
        if dtype is None:
            dtype = '<U15'
        coords = {
            'index': np.sort(np.array(list(map(str, range(n_rows))), dtype=dtype)),
            'columns': np.sort(np.array(list(map(str, range(n_cols))), dtype=dtype))
        }

    return xarray.DataArray(
        np.random.rand(n_rows, n_cols),
        dims=['index', 'columns'],
        coords=coords
    )


class TestTensorClient:
    @pytest.fixture(autouse=True)
    def test_setup_tests(self, tmpdir):
        path = tmpdir.strpath
        self.tensor_client = TensorClient(
            base_map=fsspec.get_mapper(path),
            synchronizer='thread'
        )
        self.arr = xarray.DataArray(
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
        self.arr2 = xarray.DataArray(
            data=np.array([
                [1, 2, 7, 4, 5],
                [2, 6, 5, 5, 6],
                [3, 3, 11, 5, 6],
                [4, 3, 10, 5, 6],
                [5, 7, 8, 5, 6],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [0, 1, 2, 3, 4], 'columns': [0, 1, 2, 3, 4]},
        )
        self.arr3 = xarray.DataArray(
            data=np.array([
                [1, 2, 7, 4, 5],
                [2, 6, 5, 5, 6],
                [3, 3, 11, 5, 6],
                [4, 3, 10, 5, 6],
                [5, 7, 8, 5, 6],
                [5, 7, 8, 5, 6],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [0, 1, 2, 3, 4, 5], 'columns': [0, 1, 2, 3, 4]},
        )

        for definition_id, data in tensors_definition.items():
            self.tensor_client.create_tensor(path=definition_id, definition=data)

        # store the data, so all the tests of append or update avoid running the test_store multiple times
        for path, data in [('data_one', self.arr), ('data_two', self.arr2), ('data_three', self.arr3)]:
            self.tensor_client.create_tensor(path=path, definition={})
            self.tensor_client.store(new_data=data, path=path)

    def test_create_tensor(self):
        pass

    def test_store(self):
        for path, data in [('data_one', self.arr), ('data_two', self.arr2), ('data_three', self.arr3)]:
            assert self.tensor_client.read(path=path).equals(data)

    def test_update(self):
        self.tensor_client.update(new_data=self.arr2, path='data_one')
        assert self.tensor_client.read(path='data_one').equals(self.arr2)

    def test_append(self):
        arr = create_dummy_array(10, 5, dtype=int)
        arr = arr.sel(
            index=(
                ~arr.coords['index'].isin(
                    self.tensor_client.read(
                        path='data_one'
                    ).coords['index']
                )
            )
        )

        for i in range(arr.sizes['index']):
            self.tensor_client.append(new_data=arr.isel(index=[i]), path='data_one')

        assert self.tensor_client.read(path='data_one').sel(arr.coords).equals(arr)
        assert self.tensor_client.read(path='data_one').sizes['index'] > arr.sizes['index']

    def test_delete_tensor(self):
        self.tensor_client.delete_tensor('data_one')
        assert not self.tensor_client.exist('data_one')

    def test_read_from_formula(self):
        self.tensor_client.create_tensor(path='data_four', definition=tensors_definition['data_four'])

        data_four = self.tensor_client.read(path='data_four')
        data_one = self.tensor_client.read(path='data_one')
        data_two = self.tensor_client.read(path='data_two')
        assert data_four.equals((data_one * data_two).rolling({'index': 3}).sum())

    def test_ffill(self):
        self.tensor_client.create_tensor(path='data_ffill', definition=tensors_definition['data_ffill'])
        self.tensor_client.store(path='data_ffill')
        assert self.tensor_client.read(
            path='data_ffill'
        ).equals(
            self.tensor_client.read(path='data_one').ffill('index')
        )

    def test_last_valid_dim(self):
        self.tensor_client.create_tensor(path='last_valid_dim', definition=tensors_definition['last_valid_dim'])
        self.tensor_client.store(path='last_valid_dim')
        assert np.array_equal(self.tensor_client.read(path='last_valid_dim').values, [2, 4, 4, 4, 4])

    def test_replace_by_last_valid(self):
        self.tensor_client.store(path='data_replace_by_last_valid')

        data_ffill = self.tensor_client.read(path='data_one').ffill(dim='index').compute()
        data_ffill.loc[[3, 4], 0] = np.nan
        assert self.tensor_client.read(path='data_replace_by_last_valid').equals(data_ffill)

    def test_append_reindex(self):
        self.tensor_client.store(path='reindex_data')
        reindex_data = self.tensor_client.read(path='reindex_data')
        assert reindex_data.sel(index=5, drop=True).equals(reindex_data.sel(index=4, drop=True))

    def test_overwrite_append_data(self):
        self.tensor_client.append(path='overwrite_append_data')

    def test_specifics_definition(self):
        self.tensor_client.store('specific_definition')
        assert self.tensor_client.read('specific_definition').equals(self.tensor_client.read('data_one') ** 2)

    def test_different_client(self):
        self.tensor_client.store(path='different_client', name='different_client', new_data={'a': 100})
        assert {'a': 100} == self.tensor_client.read(path='different_client', name='different_client')


if __name__ == "__main__":
    test = TestTensorClient()
    test.test_setup_tests()
    # test.test_add_definition()
    # test.test_store()
    # .test_update()
    # test.test_append()
    # test.test_backup()
    # test.test_read_from_formula()
    # test.test_ffill()
    # test.test_replace_by_last_valid()
    # test.test_last_valid_dim()
    # test.test_reindex()
    # test.test_overwrite_append_data()
    # test.test_specifics_definition()
    # test.test_different_client()
