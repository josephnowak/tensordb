import os
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from tensordb.storages.lock import PrefixLock
from zarr.storage import FSStore


class Mapping(MutableMapping):
    def __init__(
            self,
            mapper: MutableMapping,
            sub_path: str = None,
            read_lock: PrefixLock = None,
            write_lock: PrefixLock = None,
            root: str = None,
            enable_sub_map: bool = True,
    ):
        self.mapper = mapper
        self.sub_path = sub_path
        self.read_lock = PrefixLock("") if read_lock is None else read_lock
        self.write_lock = self.read_lock if write_lock is None else write_lock
        self._root = root
        self.enable_sub_map = enable_sub_map and hasattr(mapper, 'fs')

    @property
    def root(self):
        if self._root is not None:
            return self._root
        root = None
        if hasattr(self.mapper, 'root'):
            root = self.mapper.root
        elif hasattr(self.mapper, 'path'):
            root = self.mapper.path
        elif hasattr(self.mapper, 'url'):
            root = self.mapper.url

        self._root = root
        return self._root

    def sub_map(self, sub_path):
        mapper = self.mapper
        root = self.root
        if self.enable_sub_map:
            if root is not None:
                root = f'{root}/{sub_path}'
            mapper = FSStore(root, fs=mapper.fs)

        sub_path = self.add_sub_path(sub_path)
        return Mapping(
            mapper=mapper,
            sub_path=sub_path,
            read_lock=self.read_lock,
            write_lock=self.write_lock,
            root=root,
            enable_sub_map=self.enable_sub_map
        )

    def add_root(self, key):
        if key is None:
            return self.root
        if self.root is None:
            return key
        return f'{self.root}/{key}'

    def add_sub_path(self, key):
        if self.enable_sub_map or self.sub_path is None:
            return key

        if key is None:
            return self.sub_path

        return f'{self.sub_path}/{key}'

    def full_path(self, key):
        return self.add_root(self.add_sub_path(key))

    def add_lock_path(self, key):
        if self.sub_path is None:
            return key
        return f'{self.sub_path}/{key}'

    def __getitem__(self, key):
        with self.read_lock[self.add_lock_path(key)]:
            return self.mapper[self.add_sub_path(key)]

    def __setitem__(self, key, value):
        with self.write_lock[self.add_lock_path(key)]:
            self.mapper[self.add_sub_path(key)] = value

    def __delitem__(self, key):
        with self.write_lock[self.add_lock_path(key)]:
            del self.mapper[self.add_sub_path(key)]

    def __iter__(self):
        for key in self.mapper:
            if self.enable_sub_map:
                yield key
            elif key.startswith(self.sub_path):
                yield key[len(self.sub_path) + 1:]

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        key = self.add_sub_path(key)
        return key in self.mapper

    def setitems(self, values):
        # Not possible to lock
        self.mapper.setitems({self.add_sub_path(k): v for k, v in values.items()})

    def getitems(self, keys, **kwargs):
        # Not possible to lock
        return self.mapper.getitems([self.add_sub_path(k) for k in keys], **kwargs)

    def delitems(self, keys, **kwargs):
        self.mapper.delitems(keys, **kwargs)

    def listdir(self, path=None):
        if hasattr(self.mapper, 'listdir'):
            return self.mapper.listdir(self.add_sub_path(path))

        sub_map = self.mapper if path is None else self.sub_map(path)
        return list(sub_map)

    def rmdir(self, path=None):
        sub_map = self.mapper if path is None else self.sub_map(path)

        total_keys = list(sub_map.keys())
        if len(total_keys) == 0:
            return

        return sub_map.delitems(total_keys)

    def info(self, path):
        return self.mapper.fs.info(self.full_path(path))

    def checksum(self, key):
        return self.mapper.fs.checksum(self.full_path(key))

    def equal_content(self, other, path, method: Literal["checksum", "content"] = "checksum"):
        if method == "checksum":
            return other.checksum(path) == self.checksum(path)

        info = self.info(path)
        other_info = other.info(path)
        if info["size"] == other_info["size"]:
            return self[path] == other[path]

        return False

    @staticmethod
    def synchronize(
            remote_map: "Mapping",
            local_map: "Mapping",
            checksum_map: "Mapping",
            to_local: bool,
            force: bool = False,
    ):
        remote_paths = set(list(remote_map.keys()))
        local_paths = set(list(local_map.keys()))
        total_paths = list(remote_paths | local_paths)

        def _move_data(path):
            if to_local:
                if path in local_paths and path not in remote_paths:
                    del local_map[path]
                    if path in checksum_map:
                        del local_map[path]
                    return

                remote_checksum = str(remote_map.checksum(path))
                if not force and path in checksum_map:
                    local_checksum = checksum_map[path].decode()
                    if local_checksum == remote_checksum:
                        return

                local_map[path] = remote_map[path]
                checksum_map[path] = remote_checksum.encode()
            else:
                if path in remote_paths and path not in local_paths:
                    del remote_paths[path]
                    return

                if not force and path in checksum_map:
                    remote_checksum = str(remote_map.checksum(path))
                    local_checksum = checksum_map[path].decode()
                    if local_checksum == remote_checksum:
                        return

                remote_map[path] = local_map[path]

        with ThreadPoolExecutor(os.cpu_count()) as p:
            list(p.map(_move_data, total_paths))

    def folders_synchronize(
            self,
            destination,
            folders,
            comparing_method: Literal["checksum", "content"],
            n_threads
    ):
        source_paths = [
            file
            for folder in folders
            for file in self.listdir(folder)
        ]
        destination_paths = [
            file
            for folder in folders
            for file in destination.listdir(folder)
        ]
        delete_paths = sorted(set(destination_paths) - set(source_paths))

        def copy_file(path):
            if path in destination and self.equal_content(destination, path, comparing_method):
                return None
            destination[path] = self[path]
            return path

        def del_file(path):
            del destination[path]

        modified_files = list(delete_paths)
        with ThreadPoolExecutor(n_threads) as p:
            modified_files.extend([
                path
                for path in p.map(copy_file, source_paths)
                if path is not None
            ])
            list(p.map(del_file, delete_paths))

        return modified_files
