# import fsspec
# import pytest
# import json
#
# from tensordb.storages import Mapping
# from loguru import logger
#
#
# @pytest.fixture(scope="function")
# def regular_mapping(tmpdir):
#     sub_path = tmpdir.strpath
#     mapping = Mapping(fsspec.get_mapper(sub_path))
#     yield mapping
#
#
# @pytest.fixture(scope="function")
# def mongo_mapping(tmpdir):
#     from zarr.storage import MongoDBStore
#     sub_path = tmpdir.strpath
#     mongo_mapper = MongoDBStore(
#         database='test',
#         collection=sub_path
#     )
#     mapping = Mapping(mongo_mapper)
#     yield mapping
#
#
# @pytest.fixture(scope="function")
# def kazoo_mapping(tmpdir):
#     from kazoo.client import KazooClient
#
#     sub_path = tmpdir.strpath
#     zk = KazooClient(hosts='localhost:8000', timeout=2)
#     zk.start()
#     mapping = Mapping(
#         fsspec.get_mapper(sub_path),
#         read_lock=zk.ReadLock,
#         write_lock=zk.WriteLock,
#     )
#     yield mapping
#
#     zk.stop()
#
#
# def test_regular_mapping(kazoo_mapping):
#     logger.info(kazoo_mapping)
#     kazoo_mapping['test'] = json.dumps({"aj": 1}).encode('utf-8')
#     raise ValueError
#
