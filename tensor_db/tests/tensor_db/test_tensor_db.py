import xarray
import numpy as np

from tensor_db import TensorDB
from tensor_db.core.utils import create_dummy_array
from tensor_db.file_handlers import ZarrStorage
from tensor_db.config.config_root_dir import TEST_DIR_TENSOR_DB


def get_default_tensor_db():
    default_settings = {
        'handler': {
            'dims': ['index', 'columns'],
            'data_handler': ZarrStorage,
        },
    }

    tensors_definition = {
        'data_one': default_settings.copy(),
        'data_two': default_settings.copy(),
        'data_three': default_settings.copy(),
        'data_four': {
            'read': {
                'personalized_method': 'read_from_formula',
            },
            'read_from_formula': {
                'formula': "(`data_one` * `data_two`).rolling({'index': 3}).sum()",
            }
        },

        'data_ffill': {
            **default_settings,
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
            **default_settings,
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
                'value': np.nan,
                'dim': 'index'
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
                'personalized_method': 'store'
            }
        }
    }

    return TensorDB(
        base_path=TEST_DIR_TENSOR_DB,
        tensors_definition=tensors_definition,
        use_env=False
    )


class TestTensorDB:
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

    def test_store(self):
        tensor_db = get_default_tensor_db()
        tensor_db.store(new_data=TestTensorDB.arr, path='data_one')
        assert tensor_db.read(path='data_one').equals(TestTensorDB.arr)

        tensor_db.store(new_data=TestTensorDB.arr2, path='data_two')
        assert tensor_db.read(path='data_two').equals(TestTensorDB.arr2)

        tensor_db.store(new_data=TestTensorDB.arr3, path='data_three')
        assert tensor_db.read(path='data_three').equals(TestTensorDB.arr3)

    def test_update(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        tensor_db.update(new_data=TestTensorDB.arr2, path='data_one')
        assert tensor_db.read(path='data_one').equals(TestTensorDB.arr2)

    def test_append(self):
        self.test_store()
        tensor_db = get_default_tensor_db()

        arr = create_dummy_array(10, 5, dtype=int)
        arr = arr.sel(
            index=(
                ~arr.coords['index'].isin(
                    tensor_db.read(
                        path='data_one'
                    ).coords['index']
                )
            )
        )

        for i in range(arr.sizes['index']):
            tensor_db.append(new_data=arr.isel(index=[i]), path='data_one')

        assert tensor_db.read(path='data_one').sel(arr.coords).equals(arr)
        assert tensor_db.read(path='data_one').sizes['index'] > arr.sizes['index']

    # def test_backup(self):
    #     tensor_db = get_default_tensor_db()
    #     tensor_db.store(new_data=TestTensorDB.arr, path='data_one')
    #
    #     handler = tensor_db._get_handler(path='data_one')
    #     assert handler.s3_handler is not None
    #     assert handler.check_modification
    #
    #     handler.backup()
    #     assert not handler.update_from_backup()
    #     assert handler.update_from_backup(force_update_from_backup=True)
    #
    #     assert tensor_db.read(path='data_one').sel(TestTensorDB.arr.coords).equals(TestTensorDB.arr)

    def test_read_from_formula(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        data_four = tensor_db.read(path='data_four')
        data_one = tensor_db.read(path='data_one')
        data_two = tensor_db.read(path='data_two')
        assert data_four.equals((data_one * data_two).rolling({'index': 3}).sum())

    def test_ffill(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        tensor_db.store(path='data_ffill')
        assert tensor_db.read(path='data_ffill').equals(tensor_db.read(path='data_one').ffill('index'))

    def test_last_valid_index(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        tensor_db.store(path='last_valid_index')
        assert np.array_equal(tensor_db.read(path='last_valid_index').values, [2, 4, 4, 4, 4])

    def test_replace_last_valid_dim(self):
        self.test_last_valid_index()
        tensor_db = get_default_tensor_db()
        tensor_db.store(path='data_replace_last_valid_dim')

        data_ffill = tensor_db.read(path='data_ffill')
        data_ffill.loc[[3, 4], 0] = np.nan
        assert tensor_db.read(path='data_replace_last_valid_dim').equals(data_ffill)

    def test_reindex(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        tensor_db.store(path='data_reindex')
        data_reindex = tensor_db.read(path='data_reindex')
        assert data_reindex.sel(index=5, drop=True).equals(data_reindex.sel(index=4, drop=True))

    def test_overwrite_append_data(self):
        self.test_store()
        tensor_db = get_default_tensor_db()
        tensor_db.append(path='overwrite_append_data')


if __name__ == "__main__":
    test = TestTensorDB()
    # test.test_store()
    # test.test_update()
    # test.test_append()
    # test.test_backup()
    # test.test_read_from_formula()
    # test.test_ffill()
    # test.test_replace_last_valid_dim()
    # test.test_last_valid_index()
    # test.test_reindex()
    test.test_overwrite_append_data()


