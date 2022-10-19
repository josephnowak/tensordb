import os

from collections.abc import MutableMapping
from typing import ContextManager
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from zarr.storage import FSStore


class NoLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Mapping(MutableMapping):
    def __init__(
            self,
            mapper: MutableMapping,
            sub_path: str = None,
            read_lock: ContextManager = None,
            write_lock: ContextManager = None,
            root: str = None,
            enable_sub_map: bool = True,
    ):
        self.mapper = mapper
        self.sub_path = sub_path
        self.read_lock = NoLock if read_lock is None else read_lock
        self.write_lock = self.read_lock if write_lock is None else write_lock
        self._root = root
        self.enable_sub_map = enable_sub_map

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
        if self.enable_sub_map and hasattr(mapper, 'fs'):
            if root is not None:
                root = f'{root}/{sub_path}'

            if isinstance(mapper, FSStore):
                mapper = FSStore(root, fs=mapper.fs)
            elif hasattr(mapper.fs, 'get_mapper'):
                mapper = mapper.fs.get_mapper(root)
            sub_path = None
        return Mapping(mapper, sub_path, self.read_lock, self.write_lock, root)

    def add_root(self, key):
        if key is None:
            return self.root
        if self.root is None:
            return key
        return f'{self.root}/{key}'

    def add_sub_path(self, key):
        if key is None:
            return self.sub_path
        if self.sub_path is None:
            return key
        return f'{self.sub_path}/{key}'

    def full_path(self, key):
        return self.add_root(self.add_sub_path(key))

    def __getitem__(self, key):
        key = self.add_sub_path(key)
        with self.read_lock(self.add_root(key)):
            return self.mapper[key]

    def __setitem__(self, key, value):
        key = self.add_sub_path(key)
        with self.write_lock(self.add_root(key)):
            self.mapper[key] = value

    def __delitem__(self, key):
        key = self.add_sub_path(key)
        with self.write_lock(self.add_root(key)):
            del self.mapper[key]

    def __iter__(self):
        for key in self.mapper:
            if self.sub_path is None or key.startswith(self.sub_path):
                yield key

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        key = self.add_sub_path(key)
        return key in self.mapper

    def listdir(self, path=None):
        if path is None:
            path = self.sub_path
        else:
            path = self.add_sub_path(path)

        if hasattr(self.mapper, 'listdir'):
            return self.mapper.listdir(path)

        if hasattr(self.mapper, 'fs') and hasattr(self.mapper.fs, 'listdir'):
            try:
                return self.mapper.fs.listdir(self.add_root(path), detail=False)
            except (FileNotFoundError,  KeyError) as e:
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
        if path is None:
            path = self.sub_path
        elif path is not None:
            path = self.add_sub_path(path)

        if hasattr(self.mapper, 'rmdir'):
            return self.mapper.rmdir(path)

        if hasattr(self.mapper, 'fs') and hasattr(self.mapper.fs, 'delete'):
            path = self.add_root(path)
            path = path if path is not None else ''
            return self.mapper.fs.delete(path, recursive=True)

        path = path if path is not None else ''
        for key in self:
            if key.startswith(path):
                del self[key]

    def modified(self, key):
        return pd.Timestamp(self.mapper.fs.modified(self.full_path(key)))

    def copy_to_mapping(self, local_map: "Mapping", after_date: pd.Timestamp):
        total_paths = list(set(list(self.keys()) + list(local_map.keys())))

        def _move_data(path):
            if path not in self:
                del local_map[path]
                return
            if self.modified(path) < after_date:
                return
            local_map[path] = self[path]

        with ThreadPoolExecutor(os.cpu_count()) as p:
            list(p.map(_move_data, total_paths))
