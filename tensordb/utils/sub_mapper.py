import fsspec
import zarr
import os

from zarr.storage import (
    listdir,
    rmdir,
    normalize_storage_path,
)

from collections.abc import MutableMapping
from loguru import logger


class SubMapping(MutableMapping):
    """
    Useful class for handle imaginary sub folder in the cases where you use RedisStore or MongoDBStore of Zarr
    which are not paths or something related
    """

    def __init__(self,
                 store: MutableMapping,
                 path: str = None,):
        self.store = store
        self.path = None
        if isinstance(store, SubMapping):
            self.store = store.store

        if issubclass(type(self.store), (fsspec.AbstractFileSystem, fsspec.FSMap)):
            if path is not None:
                self.store = fsspec.FSMap(root=os.path.join(self.store.root, path), fs=self.store.fs)

        elif issubclass(type(self.store), zarr.storage.FSStore):
            if path is not None:
                self.store = fsspec.FSMap(root=os.path.join(self.store.map.root, path), fs=self.store.fs)
        else:
            self.path = path

    def getitems(self, keys, **kwargs):
        if self.path is not None:
            keys = [os.path.join(self.path, key) for key in keys]
        return self.store.getitems(keys, **kwargs)

    def __getitem__(self, key):
        if self.path is not None:
            key = os.path.join(self.path, key)
        return self.store[key]

    def setitems(self, values):
        if self.path is not None:
            values = {os.path.join(self.path, key): val for key, val in values.items()}
        self.store.setitems(values)

    def __setitem__(self, key, value):
        if self.path is not None:
            key = os.path.join(self.path, key)
        self.store[key] = value

    def __delitem__(self, key):
        if self.path is not None:
            key = os.path.join(self.path, key)
        del self.store[key]

    def __contains__(self, key):
        if self.path is not None:
            key = os.path.join(self.path, key)
        return key in self.store

    def __eq__(self, other):
        return self.store == other

    def keys(self):
        if self.path is not None:
            if hasattr(self.store, 'find'):
                return (x for x in self.store.find(self.path))

            if hasattr(self.store, '__iter__'):
                return (k[len(self.path) + 1:] for k in self.store if self.path == k[:len(self.path)])

            return (k[len(self.path) + 1:] for k in self.store.keys() if self.path == k[:len(self.path)])

        return self.store.keys()

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self.keys())

    def dir_path(self, path=None):
        if self.path is not None:
            path = self.path if path is None else os.path.join(self.path, path)
        return self.store.dir_path(path)

    def listdir(self, path=None):
        if self.path is not None:
            path = self.path if path is None else os.path.join(self.path, path)
        return listdir(self.store, path)

    def rmdir(self, path=None):
        if self.path is not None:
            path = self.path if path is None else os.path.join(self.path, path)
        rmdir(self.store, path)

    def getsize(self, path=None):
        if self.path is not None:
            path = self.path if path is None else os.path.join(self.path, path)
        return self.store.getsize(path)

    def clear(self):
        raise NotImplemented


