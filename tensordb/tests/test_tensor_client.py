import fsspec
import xarray as xr
import numpy as np
import pytest

from loguru import logger
from dask.distributed import Client

from tensordb import TensorClient
from tensordb.tensor_definition import TensorDefinition

# TODO: Add more tests that validate the internal behaviour of the storage settings


def create_dummy_array(n_rows, n_cols, coords=None, dtype=None) -> xr.DataArray:
    coords = coords
    if coords is None:
        dtype = dtype
        if dtype is None:
            dtype = '<U15'
        coords = {
            'index': np.sort(np.array(list(map(str, range(n_rows))), dtype=dtype)),
            'columns': np.sort(np.array(list(map(str, range(n_cols))), dtype=dtype))
        }

    return xr.DataArray(
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
        self.arr = xr.DataArray(
            data=np.array([
                [1, 2, 7, 4, np.nan],
                [np.nan, 3, 5, np.nan, 6],
                [3, 3, np.nan, np.nan, 6],
                [np.nan, 3, 10, np.nan, 6],
                [np.nan, 7, 8, 5, 6],
            ], dtype=float),
            dims=['index', 'columns'],
            coords={'index': [0, 1, 2, 3, 4], 'columns': [0, 1, 2, 3, 4]},
        )
        self.arr2 = xr.DataArray(
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
        self.arr3 = xr.DataArray(
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

        # store the data, so all the tests of append or update avoid running the test_store multiple times
        for path, data in [('data_one', self.arr), ('data_two', self.arr2), ('data_three', self.arr3)]:
            definition = TensorDefinition(
                path=path,
                definition={}
            )
            self.tensor_client.create_tensor(definition=definition)
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
        definition = TensorDefinition(
            path='data_four',
            definition={
                'read': {
                    'substitute_method': 'read_from_formula',
                },
                'read_from_formula': {
                    'formula': "new_data = (`data_one` * `data_two`).rolling({'index': 3}).sum()",
                    'use_exec': True
                }
            }
        )
        self.tensor_client.create_tensor(definition=definition)

        data_four = self.tensor_client.read(path='data_four')
        data_one = self.tensor_client.read(path='data_one')
        data_two = self.tensor_client.read(path='data_two')
        assert data_four.equals((data_one * data_two).rolling({'index': 3}).sum())

    def test_ffill(self):
        definition = TensorDefinition(
            path='data_ffill',
            definition={
                'store': {
                    'data_transformation': [
                        {'method_name': 'read_from_formula'},
                        {'method_name': 'ffill'}
                    ],
                },
                'read_from_formula': {
                    'formula': "`data_one`.chunk({'index': 3, 'columns': 2})",
                },
                'ffill': {
                    'dim': 'index',
                    'limit': 2
                }
            }
        )
        self.tensor_client.create_tensor(definition=definition)
        self.tensor_client.store(path='data_ffill')
        assert self.tensor_client.read(
            path='data_ffill'
        ).equals(
            self.tensor_client.read(path='data_one').ffill('index', limit=2)
        )

    def test_specifics_definition(self):
        definition = TensorDefinition(
            path='specific_definition',
            definition={
                'store': {
                    'data_transformation': [
                        {'method_name': 'read_from_formula', 'parameters': {'formula': "`data_one`"}},
                        {'method_name': 'read_from_formula', 'parameters': {'formula': "new_data * `data_one`"}},
                    ],
                },
            }
        )
        self.tensor_client.create_tensor(definition=definition)
        self.tensor_client.store('specific_definition')
        assert self.tensor_client.read('specific_definition').equals(self.tensor_client.read('data_one') ** 2)

    def test_different_client(self):
        definition = TensorDefinition(
            path='different_client',
            storage={
                'storage_name': 'json_storage'
            },
            definition={}
        )
        self.tensor_client.create_tensor(definition=definition)
        self.tensor_client.store(path='different_client', name='different_client', new_data={'a': 100})
        assert {'a': 100} == self.tensor_client.read(path='different_client', name='different_client')

    def test_exec_on_dag_order(self):
        definitions = [
            TensorDefinition(
                path='0',
                definition={
                    'store': {
                        'data_transformation': [
                            {'method_name': 'read_from_formula', 'parameters': {'formula': "`data_one`"}}
                        ]
                    }
                },
                dag={'depends': []}
            ),
            TensorDefinition(
                path='1',
                definition={
                    'store': {
                        'data_transformation': [
                            {'method_name': 'read_from_formula', 'parameters': {'formula': "`0` * 2"}}
                        ]
                    }
                },
                dag={'depends': ['0']}
            ),
            TensorDefinition(
                path='2',
                definition={
                    'store': {
                        'data_transformation': [
                            {'method_name': 'read_from_formula', 'parameters': {'formula': "`0` + 1"}}
                        ]
                    }
                },
                dag={'depends': ['0']}
            ),
            TensorDefinition(
                path='3',
                definition={
                    'store': {
                        'data_transformation': [
                            {'method_name': 'read_from_formula', 'parameters': {'formula': "`1` + `2`"}}
                        ]
                    }
                },
                dag={'depends': ['2', '1']}
            ),
        ]
        for definition in definitions:
            self.tensor_client.create_tensor(definition)

        client = Client()

        self.tensor_client.exec_on_dag_order(
            client=client,
            method='store'
        )
        assert self.tensor_client.read('0').equals(self.arr)
        assert self.tensor_client.read('1').equals(self.arr * 2)
        assert self.tensor_client.read('2').equals(self.arr + 1)
        assert self.tensor_client.read('3').equals(self.arr + 1 + self.arr * 2)


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
