import xarray
import numpy as np
import os
import shutil
import fsspec
import json

from loguru import logger
from time import time

from tensordb.file_handlers import JsonStorage
from tensordb.core.utils import compare_dataset
from tensordb.config.config_root_dir import TEST_DIR_JSON


def get_default_json_storage():
    return JsonStorage(
        local_base_map=fsspec.get_mapper(TEST_DIR_JSON),
        backup_base_map=fsspec.get_mapper(TEST_DIR_JSON + '/backup'),
        path='json_test',
    )


class TestJsonStorage:
    dummy_data = {'a': 0, '1': 2, 'c': {'e': 10}}

    def test_store_data(self):
        a = get_default_json_storage()
        a.store(name='first/tensor_metadata', new_data=TestJsonStorage.dummy_data)
        assert TestJsonStorage.dummy_data == a.read('first/tensor_metadata')

    def test_upsert_data(self):
        self.test_store_data()
        a = get_default_json_storage()
        upsert_d = {'b': 5, 'g': [10, 12]}
        a.upsert(name='first/tensor_metadata', new_data=upsert_d)
        assert a.read('first/tensor_metadata') == {**TestJsonStorage.dummy_data, **upsert_d}


if __name__ == "__main__":
    test = TestJsonStorage()
    test.test_store_data()
    test.test_upsert_data()
