import os
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor

from zarr.storage import FSStore

from tensordb.storages.lock import PrefixLock, NoLock


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
        self.read_lock = PrefixLock("", NoLock) if read_lock is None else read_lock
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

            if isinstance(mapper, FSStore):
                mapper = FSStore(root, fs=mapper.fs)
            else:
                mapper = mapper.fs.get_mapper(root)

        sub_path = self.add_sub_path(sub_path)
        return Mapping(mapper, sub_path, self.read_lock, self.write_lock, root, self.enable_sub_map)

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
        with self.read_lock.get_lock(self.add_lock_path(key)):
            return self.mapper[self.add_sub_path(key)]

    def __setitem__(self, key, value):
        with self.write_lock.get_lock(self.add_lock_path(key)):
            self.mapper[self.add_sub_path(key)] = value

    def __delitem__(self, key):
        with self.write_lock.get_lock(self.add_lock_path(key)):
            del self.mapper[self.add_sub_path(key)]

    def __iter__(self):
        for key in self.mapper:
            if self.enable_sub_map or key.startswith(self.sub_path):
                yield key

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        key = self.add_sub_path(key)
        return key in self.mapper

    def listdir(self, path=None):
        path = self.add_sub_path(path)

        if hasattr(self.mapper, 'listdir'):
            return self.mapper.listdir(path)

        if hasattr(self.mapper, 'fs') and hasattr(self.mapper.fs, 'listdir'):
            try:
                return self.mapper.fs.listdir(self.add_root(path), detail=False)
            except (FileNotFoundError, KeyError) as e:
                return []

        path = '' if path is None else path
        children = set()
        for key in self:
            if key.startswith(path) and len(key) > len(path):
                suffix = key[len(path):]
                child = suffix.split('/')[0]
                children.add(child)
        return sorted(children)

    def rmdir(self, path=None):
        path = self.add_sub_path(path)
        total_keys = list(self.keys())
        if len(total_keys) == 0:
            return

        if hasattr(self.mapper, 'rmdir'):
            return self.mapper.rmdir(path)

        if hasattr(self.mapper, 'fs') and hasattr(self.mapper.fs, 'delete'):
            path = self.add_root(path)
            path = path if path is not None else ''
            return self.mapper.fs.delete(path, recursive=True)

        path = path if path is not None else ''
        for key in total_keys:
            if key.startswith(path):
                del self[key]

    def checksum(self, key):
        return self.mapper.fs.checksum(self.full_path(key))

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
