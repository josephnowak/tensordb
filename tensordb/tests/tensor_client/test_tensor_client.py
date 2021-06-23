import fsspec
import xarray
import numpy as np

from loguru import logger

from tensordb import TensorClient
from tensordb.core.utils import create_dummy_array
from tensordb.config.config_root_dir import TEST_DIR_TENSOR_CLIENT


tensors_definition = {
    'data_one': {},
    'data_two': {},
    'data_three': {},
    'data_four': {
        'read': {
            'customized_method': 'read_from_formula',
        },
        'read_from_formula': {
            'formula': "new_data = (`data_one` * `data_two`).rolling({'index': 3}).sum()",
            'use_exec': True
        }
    },

    'data_ffill': {
        'store': {
            'data_methods': ['read_from_formula', 'ffill'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'ffill': {
            'dim': 'index'
        }
    },
    'data_replace_last_valid_dim': {
        'store': {
            'data_methods': ['read_from_formula', 'ffill', 'replace_last_valid_dim'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'ffill': {
            'dim': 'index'
        },
        'replace_last_valid_dim': {
            'replace_path': 'last_valid_index',
            'dim': 'index',
            'calculate_last_valid': False
        }
    },
    'last_valid_index': {
        'store': {
            'data_methods': ['read_from_formula', 'last_valid_dim'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'last_valid_dim': {
            'dim': "index",
        }
    },
    'data_reindex': {
        'store': {
            'data_methods': ['read_from_formula', 'reindex'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'reindex': {
            'coords_to_reindex': ["index"],
            'reindex_path': 'data_three',
            'method_fill_value': 'ffill'
        }
    },
    'overwrite_append_data': {
        'store': {
            'data_methods': ['read_from_formula'],
        },
        'read_from_formula': {
            'formula': "`data_one`",
        },
        'append': {
            'customized_method': 'store'
        }
    },
    'specific_definition': {
        'store': {
            'data_methods': [
                ['read_from_formula', {'formula': "`data_one`"}],
                ['read_from_formula', {'formula': "new_data * `data_one`"}],
            ],
        },
    },
    'different_client': {
        'handler': {
            'data_handler': 'json_storage'
        }
    }
}


def get_default_tensor_client():
    tensor_client = TensorClient(
        local_base_map=fsspec.get_mapper(TEST_DIR_TENSOR_CLIENT),
        backup_base_map=fsspec.get_mapper(TEST_DIR_TENSOR_CLIENT + '/backup'),
        synchronizer_definitions='thread'
    )

    return tensor_client


class TestTensorClient:
    """
    TODO: All the tests has dependencies with others, so probably should be good idea use pytest-order to establish
        an order between the tests, using this we can avoid calling some test from another tests
    """

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

    arr2 = xarray.DataArray(
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
    arr3 = xarray.DataArray(
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

    def test_add_tensor_definition(self):
        tensor_client = get_default_tensor_client()
        for tensor_id, data in tensors_definition.items():
            tensor_client.add_tensor_definition(tensor_id=tensor_id, new_data=data)
            assert tensors_definition[tensor_id] == tensor_client.get_tensor_definition(tensor_id)

    def test_store(self):
        tensor_client = get_default_tensor_client()

        tensor_client.create_tensor(path='data_one', tensor_definition='data_one')
        tensor_client.store(new_data=TestTensorClient.arr, path='data_one', synchronizer='thread')
        assert tensor_client.read(path='data_one').equals(TestTensorClient.arr)

        tensor_client.create_tensor(path='data_two', tensor_definition='data_two')
        tensor_client.store(new_data=TestTensorClient.arr2, path='data_two', synchronizer='thread')
        assert tensor_client.read(path='data_two').equals(TestTensorClient.arr2)

        tensor_client.create_tensor(path='data_three', tensor_definition='data_three')
        tensor_client.store(new_data=TestTensorClient.arr3, path='data_three', synchronizer='thread')
        assert tensor_client.read(path='data_three').equals(TestTensorClient.arr3)

    def test_update(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.update(new_data=TestTensorClient.arr2, path='data_one', synchronizer='thread')
        assert tensor_client.read(path='data_one').equals(TestTensorClient.arr2)

    def test_append(self):
        self.test_store()
        tensor_client = get_default_tensor_client()

        arr = create_dummy_array(10, 5, dtype=int)
        arr = arr.sel(
            index=(
                ~arr.coords['index'].isin(
                    tensor_client.read(
                        path='data_one'
                    ).coords['index']
                )
            )
        )

        for i in range(arr.sizes['index']):
            tensor_client.append(new_data=arr.isel(index=[i]), path='data_one')

        assert tensor_client.read(path='data_one').sel(arr.coords).equals(arr)
        assert tensor_client.read(path='data_one').sizes['index'] > arr.sizes['index']

    def test_backup(self):
        tensor_client = get_default_tensor_client()

        tensor_client.create_tensor(path='data_one', tensor_definition=tensors_definition['data_one'])
        tensor_client.store(new_data=TestTensorClient.arr, path='data_one')

        handler = tensor_client._get_handler(path='data_one')

        handler.backup()
        assert not handler.update_from_backup()
        assert handler.update_from_backup(force_update_from_backup=True)

        assert tensor_client.read(path='data_one').sel(TestTensorClient.arr.coords).equals(TestTensorClient.arr)

    def test_read_from_formula(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='data_four', tensor_definition='data_four')

        data_four = tensor_client.read(path='data_four')
        data_one = tensor_client.read(path='data_one')
        data_two = tensor_client.read(path='data_two')
        assert data_four.equals((data_one * data_two).rolling({'index': 3}).sum())

    def test_ffill(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='data_ffill', tensor_definition='data_ffill')
        tensor_client.store(path='data_ffill')
        assert tensor_client.read(path='data_ffill').equals(tensor_client.read(path='data_one').ffill('index'))

    def test_last_valid_index(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='last_valid_index', tensor_definition='last_valid_index')
        tensor_client.store(path='last_valid_index')
        assert np.array_equal(tensor_client.read(path='last_valid_index').values, [2, 4, 4, 4, 4])

    def test_replace_last_valid_dim(self):
        self.test_last_valid_index()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='data_replace_last_valid_dim', tensor_definition='data_replace_last_valid_dim')
        tensor_client.store(path='data_replace_last_valid_dim')

        data_ffill = tensor_client.read(path='data_ffill')
        data_ffill.loc[[3, 4], 0] = np.nan
        assert tensor_client.read(path='data_replace_last_valid_dim').equals(data_ffill)

    def test_reindex(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='data_reindex', tensor_definition='data_reindex')
        tensor_client.store(path='data_reindex')
        data_reindex = tensor_client.read(path='data_reindex')
        assert data_reindex.sel(index=5, drop=True).equals(data_reindex.sel(index=4, drop=True))

    def test_overwrite_append_data(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='overwrite_append_data', tensor_definition='overwrite_append_data')
        tensor_client.append(path='overwrite_append_data')

    def test_specifics_definition(self):
        self.test_store()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='specific_definition', tensor_definition='specific_definition')
        tensor_client.store('specific_definition')
        assert tensor_client.read('specific_definition').equals(tensor_client.read('data_one') ** 2)

    def test_different_client(self):
        self.test_add_tensor_definition()
        tensor_client = get_default_tensor_client()
        tensor_client.create_tensor(path='different_client', tensor_definition='different_client')
        tensor_client.store(path='different_client', name='different_client', new_data={'a': 100})
        assert {'a': 100} == tensor_client.read(path='different_client', name='different_client')


if __name__ == "__main__":
    test = TestTensorClient()
    # test.test_add_tensor_definition()
    # test.test_store()
    test.test_update()
    # test.test_append()
    # test.test_backup()
    # test.test_read_from_formula()
    # test.test_ffill()
    # test.test_replace_last_valid_dim()
    # test.test_last_valid_index()
    # test.test_reindex()
    # test.test_overwrite_append_data()
    # test.test_specifics_definition()
    # test.test_different_client()
