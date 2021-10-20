import xarray
import orjson

from typing import Dict

from tensordb.storages.base_storage import BaseStorage


class JsonStorage(BaseStorage):

    """
    This class was created with the idea of simplify how the tensor client store the definitions.
    Every path is converted into a name replacing the "/" by "_" with the idea of create unique names and allow
    to identify every tensor without the use of folders
    """

    default_chracter = "."

    def store(self, name: str, new_data: Dict):
        new_name = name.replace('\\', '/').replace('/', self.default_chracter)
        self.base_map[new_name] = orjson.dumps(new_data)

    def append(self, name: str, new_data: Dict):
        raise NotImplemented('Use upsert')

    def update(self, name: str, new_data: Dict):
        raise NotImplemented('Use upsert')

    def upsert(self, name: str, new_data: Dict):
        new_name = name.replace('\\', '/').replace('/', self.default_chracter)
        d = orjson.loads(self.base_map[new_name])
        d.update(new_data)
        self.store(name=name, new_data=d)

    def read(self, name: str) -> xarray.DataArray:
        new_name = name.replace('\\', '/').replace('/', self.default_chracter)
        return orjson.loads(self.base_map[new_name])

    def exist(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', self.default_chracter)
        return new_name in self.base_map

    def delete_file(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', self.default_chracter)
        del self.base_map[new_name]

    @classmethod
    def get_original_path(self, name):
        return name.replace(self.default_chracter, '/')




