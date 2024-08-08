import abc
import asyncio
import datetime
import os
from abc import ABC
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Any
import pandas as pd

import orjson

from tensordb.zarr_v3.abc.store import Store
from tensordb.zarr_v3.store.remote import RemoteStore
from tensordb.zarr_v3.abc.store import Store
from tensordb.zarr_v3.buffer import Buffer, BufferPrototype
from tensordb.zarr_v3.common import OpenMode
from tensordb.zarr_v3.store.core import _dereference_path
from tensordb.sync import sync_iter, sync
from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable
from tensordb.zarr_v3.common import BytesLike, OpenMode
from tensordb.zarr_v3.codecs import BytesCodec
from tensordb.zarr_v3.buffer import default_buffer_prototype
from tensordb.config import settings
from pydantic import BaseModel
import fsspec


import fsspec
from fsspec.asyn import AsyncFileSystem

class BaseBranchStorage(Store, MutableMapping, abc.ABC):
    pass


class GarbageCollector(BaseModel):
    remove_after_days: int = 252
    merge_transactions: bool = True


class Repository(BaseModel):
    name: str
    path: str
    default_branch: str = "main"
    description: str
    details: dict
    isolation: Literal["uncommited", "commited", "snapshot"] = "snapshot"
    consistency: Literal["ignore", "detect"] = "ignore"
    garbage_collector: GarbageCollector = None


class Branch(BaseModel):
    branch: str
    dependencies: set | None
    creation_date: datetime.datetime = datetime.datetime.utcnow()


class Transaction(BaseModel):
    id: str
    open_date: datetime.datetime = datetime.datetime.utcnow()
    close_date: datetime.datetime = None


class TransactionFile(BaseModel):
    path: str
    transaction_path: str
    transaction_id: str
    transaction_date: datetime.datetime
    write_date: datetime.datetime
    checksum: str


class Path(str):
    def __truediv__(self, other):
        return self + "/" + other


class BranchFS(AsyncFileSystem):

    def __init__(
            self,
            fs: AsyncFileSystem,
            repo: str,
            branch: str,
            transaction_id: str = None,
    ):
        super().__init__()
        self.fs: AsyncFileSystem = fs
        self._repo = Path(repo)
        self._branch = Path(branch)
        self._transaction_id = Path(transaction_id)
        self._transaction_path = Path(settings.TRANSACTION_PATH)
        self._status_path = Path(settings.TRANSACTION_STATUS_PATH)
        self._metadata_path = Path(settings.TRANSACTION_METADATA_PATH)
        self._data_path = Path(settings.TRANSACTION_DATA_PATH)
        self._branch_path = Path(settings.BRANCH_PATH)

    @staticmethod
    def create_repository(fs: AsyncFileSystem, repo: str, description: dict, default_branch: str = "main"):
        fs.pipe_file()

    def create_branch(self, repo: str, branch: str, parent_branch: str):
        branch_details = self.get_branch(branch)
        if branch_details is not None:
            raise KeyError(
                f"The branch: {branch} already exists on the repository {repo}"
            )

        parent_branch_details = self.get_branch(parent_branch)
        if parent_branch_details is None:
            raise KeyError(f"The parent branch {parent_branch} does not exist")

        branch = Branch(
            branch=branch,
            parent_branch=parent_branch
        )
        self.set(
            self.get_branch_path(branch),
            orjson.dumps(branch.model_dump()),
            use_transaction=False
        )

    def get_branch_path(self, branch: str):
        return self._repo / settings.BRANCH_PATH / Path(branch)

    def get_open_transactions(self):
        prefix = self._repo / self._branch / self._transaction_path / self._status_path / Path("open")
        self.list_prefix(prefix, use_transaction=False)

    def get_branch(self, branch: str) -> Branch | None:
        branch_path = self.get_branch_path(branch)
        try:
            return Branch(
                **orjson.loads(
                    sync(self.get(key=branch_path, use_transaction=False))
                )
            )
        except KeyError:
            return None

    def get_open_transactions(self):
        prefix = self._repo + "/" + self._branch + "/" + settings.METADATA_PATH + "/" + path
        return self.list_prefix()

    def search_key(self, key, use_transaction: bool = True) -> TransactionFile:
        key = Path(key)
        if not use_transaction:
            return self._repo / self._branch / key

        prefix = self._repo / self._branch / self._metadata_path / key
        paths = sync_iter(self.list_prefix(prefix, use_transaction=False))
        paths = sorted(paths)
        # All the files on the folder are the metadata created from the different transactions
        # All of them are group in the following way: {transaction_date}/{transaction_id}/{write_date}/{checksum}
        last_path = paths[-1]
        transaction_file = last_path.split("/")[-4:]
        transaction_path = self._repo + "/" + self._branch + "/" + settings.TRANSACTION_PATH + "/" + key
        transaction_file = TransactionFile(
            path=key,
            transaction_path=transaction_path,
            transaction_id=transaction_file[1],
            transaction_date=pd.Timestamp(transaction_file[0]),
            write_date=pd.Timestamp(transaction_file[2]),
            checksum=transaction_file[3],
        )
        return transaction_file

    async def _pipe_file(self, path, data, chunksize=50 * 2 ** 20, **kwargs):
        self.fs._pipe_file(path)

    async def _exists(self, path):
        pass