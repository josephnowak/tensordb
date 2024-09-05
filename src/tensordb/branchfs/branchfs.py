import abc
import asyncio
import datetime
import os
from abc import ABC
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Any
import pandas as pd

import json

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
from pydantic import BaseModel, Field, computed_field
import json
import fsspec


import fsspec
from fsspec.asyn import AsyncFileSystem


class Path(str):
    def __truediv__(self, other):
        return self + "/" + other


class BaseLock:
    def __init__(self, path: str, *args, **kwargs):
        self.path = path
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class NoLock(BaseLock):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BaseBranchStorage(Store, MutableMapping, abc.ABC):
    pass


class GarbageCollector(BaseModel):
    transaction_branch_expires_in: int = Field(
        description="Number of days after which an open transaction branch can be considered as expired "
        "which means that it can be cleaned on the vacuum process.",
    )
    vacuum_transactions_older_than_n_days: int = Field(
        description="Every transaction older than that number of days is going to be "
        "deleted if possible, which means that you are no longer going to be able "
        "to rollback to that point on the history for a certain branch.\n\n"
        "For more information about this process please read the docs of the vacuum method"
    )
    vacuum_deleted_files_older_than_n_days: int = Field(
        description="The deletion process only creates a file with a delete operation "
        "which means that all the previous files are kept on the file system "
        "until the vacuum method is called and they get older than the "
        "number of days specified on this variable, "
    )


class Repository(BaseModel):
    name: str = Field(description="Name of the repository.")
    default_branch_name: str = Field(
        "main",
        description="The default branch is created together with the repository, "
        "this branch can not be deleted, unless the repository is deleted.",
    )
    description: str = Field(
        "My first repository using BranchFS",
        description="Description of the repository.",
    )
    details: dict = Field({}, description="Additional metadata of the repository.")
    isolation: Literal["transaction_branch"] = Field(
        default="transaction_branch",
        description="BranchFS always guaranty an isolation based on the transaction branches. "
        "which means that every read on those branches is going to use the data that the parent "
        "was able to read at the moment of its creation unless the file was modified in the "
        "transaction branch (this can be seen as a snapshot isolation). \n\n"
        "Probably in the future some additional options can be added.",
    )
    consistency_method: Literal["ignore", "optimistic"] = Field(
        "optimistic",
        description="The consistency method indicates if the system must check or not any "
        "inconsistency during the merge process. If the optimistic method is selected "
        "then the system is going to check if there have been other mergers "
        "executed before yours, in whose case the system is going to validate "
        "if there are files modified on those merge that were also modified "
        "on yours, if that happens, then a proper error is raised.\n\n"
        "https://www.geeksforgeeks.org/concurrency-control-in-distributed-transactions/",
    )
    garbage_collector: GarbageCollector = Field(
        description="The garbage collector defines the rules for running the vacuum "
        "and cleaning the expired transaction branches",
    )
    branch_save_logic: Literal["empty_file", "ignore", "empty_dir"] = Field(
        "empty_file",
        description="BranchFS saves all files modifications of all branches of the same repository "
        "in the path of the file, which means that a file with a path "
        "'level1/level2/file.csv' is stored as 'level1/level2/file/long-metadata/checksum' "
        "this makes impossible to get all the files modified on a branch without listing "
        "everything, for that reason, BranchFS offers the option to store on a separate folder "
        "an empty file or an empty dir to keep track of all the modifications.\n\n"
        "It is important to highlight that an empty dir or file is going to add an overhead "
        "on every write that you do, but it is going to reduce the time that it "
        "takes to merge in certain cases. Also certain file system do not support the empty dir"
        "option, for example s3fs https://github.com/fsspec/s3fs/issues/401.",
    )
    file_metadata_sep: str = Field(
        "/",
        description="This character is used to separate the metadata that BranchFS adds "
        "to every file path for its internal use. It is recommended to use '/' as the default "
        "because certain file system do not support long file names",
    )
    file_checksum_func: Literal["crc32c", "ignore"] = Field(
        "crc32c",
        description="BranchFS always keep track of certain "
        "metadata of the files using an internal path "
        "one of the part of the metadata is the "
        "checksum which can be used to avoid unnecessary writes",
    )

    def join_paths(self, *args):
        return f"{self.file_metadata_sep}".join(args)

    def path(self):
        return self.join_paths(settings.REPOSITORY_PATH, f"{self.name}.json")

    def save(self, fs: AsyncFileSystem, ):
        repository_path = self.path()
        if fs.exists(repository_path):
            raise KeyError(
                f"The repository name {self.name} already exists on the folder "
                f"{settings.REPOSITORY_PATH}. "
                f"Please use a unique name to identify your new repository."
            )
        fs.pipe_file(repository_path, json.dumps(self.model_dump()))


class MergeHistory(BaseModel):
    branch_name: str
    merger_date: datetime.datetime = Field(
        description="Date at which both branches were merged"
    )


class Branch(BaseModel):
    branch_type: Literal["standard", "transaction"] = Field(
        "standard",
        description="The standard branches only allows a read mode, "
                    "to be able to write on them you need to create a transaction branch from it "
                    "and merge it, or merge another standard branch with it.",
    )
    group: str = Field(
        default=None,
        description="In the standard branches the group is always going to be equal to the name, "
        "while in transaction ones the group is going to be the same than the parent branch."
        "This allows to group the transactions branches inside the standard ones "
        "which allows us to avoid saving the transaction branch on the merged_branches attributes",
    )
    name: str = Field(description="Name of the branch")
    merged_branches: list[MergeHistory] = Field(
        [],
        description="The merged branches are the historical merge that has been done on "
        "this branch. The transaction branches are not maintained in this list to avoid "
        "adding a new element everytime there is a modification.\n\n"
        "The first element of this list is always the parent branch",
    )
    kvbranch: dict[str, datetime.datetime] = Field(
        {},
        description="The kvbranch is a data structured used to speed up the search of "
        "the files (taking into account the datetime) that this branch has access to.\n\n"
        "Every time a merge is effected this kvbranch is combined with the one of "
        "the other branch, and if two keys match then the most recent date is maintained. \n\n"
        "The whole algorithm is described on the readme of the project.",
    )

    @computed_field
    def group(self):
        return self.name

    def branch_path(self, repository: Repository):
        return repository.join_paths(
            settings.BRANCH_FOLDER,
            repository.name,
            self.group,
            settings.BRANCH_FOLDER,
            f"{repository.default_branch_name}.json"
        )

    def merge_path(self, repository: Repository):
        pass

    def save(self, fs: AsyncFileSystem, repository: Repository):
        branch_path = 2
        try:
            fs.pipe_file(branch_path, json.dumps(self.model_dump()))
        except Exception as e:
            # In case of any error during the creation of the branch delete the repository
            # there should not exist a repo without a default branch
            fs.delete(branch_path)
            raise e


class FileMetadata(BaseModel):
    file_path: str
    group_name: str
    branch_name: str
    operation: Literal["write", "delete"]
    write_date: datetime.datetime
    checksum: str


class BranchFS(AsyncFileSystem):
    def __init__(
        self,
        fs: AsyncFileSystem,
        repository_name: str,
        branch_name: str,
        lock: BaseLock = None
    ):
        super().__init__()
        self.fs: AsyncFileSystem = fs
        self.lock = lock
        if self.lock is None:
            self.lock = NoLock

        self._repo = self.get_repository(self.fs, repository_name)
        self._branch = self.get_branch(branch_name)
        self._transaction_path = Path(settings.TRANSACTION_PATH)
        self._status_path = Path(settings.TRANSACTION_STATUS_PATH)
        self._metadata_path = Path(settings.TRANSACTION_METADATA_PATH)
        self._data_path = Path(settings.TRANSACTION_DATA_PATH)
        self._branch_path = Path(settings.BRANCH_PATH)

    @staticmethod
    def create_repository(fs: AsyncFileSystem, repository: Repository) -> Branch:
        repository.save(fs)
        try:
            branch_path = repository.join_paths(
                base_path, settings.BRANCH_FOLDER, f"{repository.default_branch_name}.json"
            )
            branch = Branch(name=repository.default_branch_name)
            fs.pipe_file(branch_path, json.dumps(branch.model_dump()))
        except Exception as e:
            # In case of any error during the creation of the branch delete the repository
            # there should not exist a repo without a default branch
            fs.delete(repository_path)
            raise e

        return branch

    @staticmethod
    def get_repository(fs: AsyncFileSystem, name: str) -> Repository:
        path = "/".join([settings.REPOSITORY_PATH, name, "repository.json"])
        return Repository(**json.loads(fs.cat_file(path)))

    def create_branch(
        self, branch_name: str, parent_branch_name: str = None, merge_lock=None
    ) -> Branch:
        branch = self.get_branch(branch_name)
        if branch is not None:
            raise KeyError(
                f"The branch {branch_name} already exists on the repository {self._repo.name}"
            )

        if parent_branch_name is None:
            # Always reload the parent branch information to be as updated as possible
            parent_branch_name = self._branch.name

        with self.lock(parent_branch_name):
            parent_branch = self.get_branch(parent_branch_name)
            if parent_branch is not None:
                raise KeyError(
                    f"The parent branch {parent_branch_name} "
                    f"has not been created on the repo {self._repo.name}"
                )

            try:
                branch = Branch(
                    name=branch_name,
                )
            except Exception as e:
                self.fs.delete(branch_name)
                raise e

        self.set(
            self.get_branch_path(branch),
            json.dumps(branch.model_dump()),
            use_transaction=False,
        )

    def get_branch_path(self, branch: str):
        return self._repo / settings.BRANCH_PATH / Path(branch)

    def get_open_transactions(self):
        prefix = (
            self._repo
            / self._branch
            / self._transaction_path
            / self._status_path
            / Path("open")
        )
        self.list_prefix(prefix, use_transaction=False)

    def get_branch(self, branch_name: str) ->  | None:
        branch_path = self.get_branch_path(branch_name)
        try:
            return StandardBranch(
                **json.loads(sync(self.get(key=branch_path, use_transaction=False)))
            )
        except KeyError:
            return None

    def get_open_transactions(self):
        prefix = (
            self._repo + "/" + self._branch + "/" + settings.METADATA_PATH + "/" + path
        )
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
        transaction_path = (
            self._repo
            + "/"
            + self._branch
            + "/"
            + settings.TRANSACTION_PATH
            + "/"
            + key
        )
        transaction_file = TransactionFile(
            path=key,
            transaction_path=transaction_path,
            transaction_id=transaction_file[1],
            transaction_date=pd.Timestamp(transaction_file[0]),
            write_date=pd.Timestamp(transaction_file[2]),
            checksum=transaction_file[3],
        )
        return transaction_file

    async def _pipe_file(self, path, data, chunksize=50 * 2**20, **kwargs):
        self.fs._pipe_file(path)

    async def _exists(self, path):
        pass
