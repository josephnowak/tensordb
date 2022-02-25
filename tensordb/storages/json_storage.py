import xarray as xr
import orjson

from typing import Dict

from tensordb.storages.base_storage import BaseStorage


class JsonStorage(BaseStorage):

    """
    This class was created with the idea of simplify how the tensor client store the definitions.
    Every path is converted into a name replacing the "/" by "_" with the idea of create unique names and allow
    to identify every tensor without the use of folders
    """

    default_character = "."

    def store(self, name: str, new_data: Dict):
        new_name = name.replace('\\', '/').replace('/', self.default_character)
        self.get_write_base_map()[new_name] = orjson.dumps(new_data)

    def append(self, name: str, new_data: Dict):
        raise NotImplemented('Use upsert')

    def update(self, name: str, new_data: Dict):
        raise NotImplemented('Use upsert')

    def upsert(self, name: str, new_data: Dict):
        new_name = name.replace('\\', '/').replace('/', self.default_character)
        d = orjson.loads(self.base_map[new_name])
        d.update(new_data)
        self.store(name=name, new_data=d)

    def read(self, name: str) -> xr.DataArray:
        new_name = name.replace('\\', '/').replace('/', self.default_character)
        return orjson.loads(self.base_map[new_name])

    def exist(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', self.default_character)
        return new_name in self.base_map

    def drop(
            self,
            coords,
            **kwargs
    ) -> xr.backends.common.AbstractWritableDataStore:
        raise NotImplementedError

    def delete_file(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', self.default_character)
        del self.base_map[new_name]

    @classmethod
    def get_original_path(self, name):
        return name.replace(self.default_character, '/')




