import fsspec
import pytest

from tensordb.storages import JsonStorage


class TestJsonStorage:
    @pytest.fixture(autouse=True)
    def setup_tests(self, tmpdir):
        sub_path = tmpdir.strpath
        self.storage = JsonStorage(
            base_map=fsspec.get_mapper(sub_path + '/json'),
            tmp_map=fsspec.get_mapper(sub_path + '/tmp'),
            path='json_storage'
        )
        self.dummy_data = {'a': 0, '1': 2, 'c': {'e': 10}}

    def test_store_data(self):
        self.storage.store(path='first/tensor_metadata', new_data=self.dummy_data)
        assert self.dummy_data == self.storage.read('first/tensor_metadata')

    def test_upsert_data(self):
        self.storage.store(path='first/tensor_metadata', new_data=self.dummy_data)
        upsert_d = {'b': 5, 'g': [10, 12]}
        self.storage.upsert(path='first/tensor_metadata', new_data=upsert_d)
        assert self.storage.read('first/tensor_metadata') == {**self.dummy_data, **upsert_d}


if __name__ == "__main__":
    test = TestJsonStorage()
    test.test_store_data()
    test.test_upsert_data()
