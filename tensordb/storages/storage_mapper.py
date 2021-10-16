import fsspec
import zarr
import orjson
import os
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, Union, List, Any
from zarr.storage import (
    listdir,
    rmdir,
    normalize_storage_path,
)
from collections.abc import MutableMapping

from tensordb.storages.lock import get_lock, BaseLock, BaseMapLock


class StorageMapper(MutableMapping):
    """
    Useful class for handle imaginary sub folder in the cases where you use Redisfs or MongoDBfs of Zarr
    which are not paths or something related
    """

    def __init__(self,
                 fs: MutableMapping,
                 mapper_synchronizer: BaseMapLock = None,
                 path: str = None,):
        self.fs = fs
        self.mapper_synchronizer = mapper_synchronizer
        if isinstance(fs, StorageMapper):
            self.fs = fs.fs

        self.path = None
        if path is not None:
            path = normalize_storage_path(path)
            if isinstance(self.fs, fsspec.FSMap):
                self.fs = fsspec.FSMap(root=self.join_paths(self.fs.root, path), fs=self.fs.fs)
            elif isinstance(self.fs, zarr.storage.FSStore):
                self.fs = fsspec.FSMap(root=self.join_paths(self.fs.map.root, path), fs=self.fs.fs)
            else:
                self.path = path

    def get_lock(self, path) -> BaseLock:
        return get_lock(self.mapper_synchronizer, path)

    @staticmethod
    def join_paths(base_path: str, path: Union[str, List[str], Dict[str, Any]]):
        if base_path is None:
            return path
        if path is None:
            return path
        if isinstance(path, str):
            return base_path + '/' + path
        if isinstance(path, list):
            return [base_path + '/' + sub_path for sub_path in path]
        if isinstance(path, dict):
            return {base_path + '/' + sub_path: val for sub_path, val in path.items()}

        raise TypeError(f'{type(path)} is not supported')

    def getitems(self, keys, **kwargs):
        keys = self.join_paths(self.path, keys)
        return self.fs.getitems(keys, **kwargs)

    def __getitem__(self, key):
        key = self.join_paths(self.path, key)
        return self.fs[key]

    def setitems(self, values):
        values = self.join_paths(self.path, values)
        self.fs.setitems(values)

    def __setitem__(self, key, value):
        key = self.join_paths(self.path, key)
        with self.get_lock(key):
            self.fs[key] = value

    def __delitem__(self, key):
        key = self.join_paths(self.path, key)
        with self.get_lock(key):
            del self.fs[key]

    def __contains__(self, key):
        key = self.join_paths(self.path, key)
        return key in self.fs

    def __eq__(self, other):
        return self.fs == other

    def keys(self):
        if self.path is not None:
            if hasattr(self.fs, 'find'):
                return (x for x in self.fs.find(self.path))

            if hasattr(self.fs, '__iter__'):
                return (k[len(self.path) + 1:] for k in self.fs if self.path == k[:len(self.path)])

            return (k[len(self.path) + 1:] for k in self.fs.keys() if self.path == k[:len(self.path)])

        return self.fs.keys()

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self.keys())

    def dir_path(self, path=None):
        path = self.join_paths(self.path, path)
        return self.fs.dir_path(path)

    def listdir(self, path=None):
        path = self.join_paths(self.path, path)
        return listdir(self.fs, path)

    def rmdir(self, path=None):
        path = self.join_paths(self.path, path)
        rmdir(self.fs, path)
        self.fs.rmdir(path)

    def delitems(self, keys):
        keys = self.join_paths(self.path, keys)
        self.fs.delitems(keys=keys)

    def getsize(self, path=None):
        path = self.join_paths(self.path, path)
        return self.fs.getsize(path)

    def clear(self):
        self.fs.clear()

    def checksum(self, path: str) -> str:
        path = self.join_paths(self.path, path)
        with self.get_lock(path):
            if hasattr(self.fs, 'checksum'):
                return str(self.fs.checksum(path))

            if hasattr(self.fs, 'fs'):
                return str(self.fs.fs.checksum(f'{self.fs.root}/{path}'))

            raise NotImplemented(f'The fs that you are using does not have a checksum method or use FSMap class')

    def checksums(self, paths: List[str]) -> Dict[str, str]:
        with ThreadPoolExecutor(min(os.cpu_count(), len(paths))) as executor:
            return dict(zip(paths, executor.map(self.checksum, tuple(paths))))

    def get_as_dict(self, path: str, default_value=None, raise_error: bool = False) -> Dict:
        if raise_error:
            return orjson.loads(self[path])
        try:
            return orjson.loads(self[path])
        except KeyError:
            return default_value

    def update_json(self, path: str, d: Union[Dict, str]):
        act_d = self.get_as_dict(path, {})
        if isinstance(d, str):
            d = self.get_as_dict(d, raise_error=True)
        act_d.update(d)
        self[path] = orjson.dumps(act_d)

    def transfer_files(self, from_path_map: 'StorageMapper', paths: Union[str, List[str]]):
        """
        Transfer the files from another file system to this one.

        :param from_path_map: File system for getting the files
        :param paths:
        :return:
        """
        paths = [paths] if isinstance(paths, str) else paths
        paths = list(set(paths))
        if len(paths) == 0:
            return

        def transfer(path):
            self[path] = from_path_map[path]

        with ThreadPoolExecutor(min(os.cpu_count(), len(paths))) as executor:
            executor.map(transfer, tuple(paths))
