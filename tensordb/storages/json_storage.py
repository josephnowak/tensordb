from typing import Dict

import orjson
import xarray as xr

from tensordb.storages.base_storage import BaseStorage


class JsonStorage(BaseStorage):
    """
    This class was created with the idea of simplify how the tensor client store the definitions.
    Every path is converted into a name replacing the "/" by "_" with the idea of create unique names and allow
    to identify every tensor without the use of folders
    """

    default_character = "."

    @classmethod
    def to_json_file_name(cls, path):
        return path.replace('\\', '/').replace('/', cls.default_character)

    def store(self, new_data: Dict, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        self.base_map[new_name] = orjson.dumps(new_data)

    def append(self, new_data: Dict, path: str = None, **kwargs):
        raise NotImplemented('Use upsert')

    def update(self, new_data: Dict, path: str = None, **kwargs):
        raise NotImplemented('Use upsert')

    def upsert(self, new_data: Dict, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        try:
            d = orjson.loads(self.base_map[new_name])
        except KeyError:
            d = {}
        d.update(new_data)
        self.store(path=path, new_data=d)

    def read(self, path: str = None) -> Dict:
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        return orjson.loads(self.base_map[new_name])

    def exist(self, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        return new_name in self.base_map

    def drop(
            self,
            coords,
            **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        raise NotImplementedError

    def delete_file(self, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        del self.base_map[new_name]

    @classmethod
    def get_original_path(cls, path):
        return path.replace(cls.default_character, '/')
