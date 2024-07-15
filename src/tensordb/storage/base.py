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


class BaseBranchStorage(Store, MutableMapping, abc.ABC):
    pass


class Branch(BaseModel):
    branch: str
    parent_branch: str | None
    creation_date: datetime.datetime = datetime.datetime.utcnow()


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

import s3fs.core

class BranchStorage(BaseBranchStorage):
    # based on FSSpec
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    def __init__(
            self,
            storage: Store,
            repo: str,
            branch: str,
            sub_path: str = None,
            transaction_id: str = None,
    ):
        super().__init__()
        self._storage = storage
        self._repo = Path(repo)
        self._branch = Path(branch)
        self._sub_path = Path(sub_path)
        self._transaction_id = Path(transaction_id)
        self._transaction_path = Path(settings.TRANSACTION_PATH)
        self._status_path = Path(settings.TRANSACTION_STATUS_PATH)
        self._metadata_path = Path(settings.TRANSACTION_METADATA_PATH)
        self._data_path = Path(settings.TRANSACTION_DATA_PATH)
        self._branch_path = Path(settings.BRANCH_PATH)

    def create_branch(self, repo: str, branch: str, parent_branch: str = None):
        branch_details = self.get_branch(branch)
        if branch_details is not None:
            raise KeyError(
                f"The branch: {branch} already exists on the repository {repo}"
            )

        if parent_branch is not None:
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

    def __str__(self) -> str:
        return self._storage.__str__()

    def __repr__(self) -> str:
        return self._storage.__repr__()

    async def get(
            self,
            key: str,
            prototype: BufferPrototype = default_buffer_prototype,
            byte_range: tuple[int | None, int | None] | None = None,
            use_transaction: bool = True
    ) -> Buffer | None:
        if not use_transaction:
            return await self._storage.get(key=key, prototype=prototype, byte_range=byte_range)
        if self._transaction_id is None:
            raise ValueError(f"There is no transaction open")

        transaction_file = self.get_transaction_file(key)
        return await self._storage.get(
            transaction_file.transaction_path, prototype=prototype, byte_range=byte_range
        )

    async def set(
            self,
            key: str,
            value: Buffer,
            byte_range: tuple[int, int] | None = None,
            use_transaction: bool = True
    ) -> None:
        if self._transaction_id is None:
            raise ValueError(f"There is no transaction open")

        try:
            return await self._storage.set(key=key, value=value, byte_range=byte_range)
        except TypeError:
            return await self._storage.set(key=key, value=value)

    async def delete(self, key: str, use_transaction: bool = True) -> None:
        return await self._storage.delete(key=key)

    async def exists(self, key: str, use_transaction: bool = True) -> bool:
        return await self._storage.exists(key=key)

    async def get_partial_values(
            self,
            prototype: BufferPrototype = None,
            key_ranges: list[tuple[str, tuple[int | None, int | None]]] = None,
            use_transaction: bool = True
    ) -> list[Buffer | None]:
        prototype = default_buffer_prototype if prototype is None else prototype
        if key_ranges is None:
            raise ValueError("The key_ranges must not be null")
        return await self._storage.get_partial_values(prototype=prototype, key_ranges=key_ranges)

    def set_partial_values(
            self,
            key_start_values: list[tuple[str, int, BytesLike]],
            use_transaction: bool = True
    ) -> None:
        raise NotImplementedError

    async def list(self, use_transaction: bool = True) -> AsyncGenerator[str, None]:
        return self._storage.list()

    async def list_dir(
            self,
            prefix: str,
            use_transaction: bool = True
    ) -> AsyncGenerator[str, None]:
        if not use_transaction:
            return self._storage.list_dir(prefix=prefix)

        prefix_ref = self._repo / self._branch / self._transaction_path / Path(prefix)

    async def list_prefix(self, prefix: str, use_transaction: bool = True) -> AsyncGenerator[str, None]:
        # if not use_transaction:
        #     return self._storage.list_prefix(prefix)
        prefix = self._repo / self._branch / self._transaction_path / self._data_path

        duplicated = set()
        for path in self._storage.list_prefix(prefix):
            yield path
        # return self._storage.list_prefix(prefix)

    def __setitem__(self, key: str, value: BytesLike):
        value = Buffer.from_bytes(value)
        sync(self.set(key=key, value=value))

    def __delitem__(self, key: str):
        sync(self.delete(key=key))

    def __getitem__(self, key: str) -> BytesLike:
        return sync(self.get(key=key)).to_bytes()

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        return sync_iter(self.list_prefix(self.sub_path))

    def __contains__(self, key):
        key = self.add_sub_path(key)
        return key in self.mapper

    def setitems(self, values: dict):
        async def _setitems():
            await asyncio.gather([self.set(key=k, value=v) for k, v in values.items()])

        sync(_setitems())

    def getitems(self, keys, **kwargs):
        # Not possible to lock
        return self.mapper.getitems([self.add_sub_path(k) for k in keys], **kwargs)

    def rmdir(self, path=None):
        sub_map = self.mapper if path is None else self.sub_map(path)

        total_keys = list(sub_map.keys())
        if len(total_keys) == 0:
            return

        return sub_map.delitems(total_keys)

    def delitems(self, keys, **kwargs):
        self.mapper.delitems(keys, **kwargs)

    def info(self, path):
        return self.mapper.fs.info(self.full_path(path))
