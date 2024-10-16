
import orjson
import xarray as xr
from pydantic.v1.utils import deep_update

from tensordb.storages.base_storage import BaseStorage


class JsonStorage(BaseStorage):
    """
    This class was created with the idea of simplify how the tensor client store the definitions.
    """

    default_character = "/"

    @classmethod
    def to_json_file_name(cls, path):
        return path.replace("\\", "/").replace("/", cls.default_character)

    def store(self, new_data: dict, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        self.base_map[new_name] = orjson.dumps(
            new_data, option=orjson.OPT_SERIALIZE_NUMPY
        )

    def append(self, new_data: dict, path: str = None, **kwargs):
        raise NotImplementedError("Use upsert")

    def update(self, new_data: dict, path: str = None, **kwargs):
        raise NotImplementedError("Use upsert")

    def upsert(self, new_data: dict, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        try:
            d = orjson.loads(self.base_map[new_name])
        except KeyError:
            d = {}
        d = deep_update(d, new_data)
        self.store(path=path, new_data=d)

    def read(self, path: str = None) -> dict:
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        try:
            return orjson.loads(self.base_map[new_name])
        except orjson.JSONDecodeError:
            # This error can be raised if there are multiple writes and reads in parallel
            return {}

    def exist(self, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        return new_name in self.base_map

    def drop(self, coords, **kwargs) -> xr.backends.common.AbstractWritableDataStore:
        raise NotImplementedError

    def delete_file(self, path: str = None, **kwargs):
        path = self.data_names if path is None else path
        new_name = self.to_json_file_name(path)
        del self.base_map[new_name]

    @classmethod
    def get_original_path(cls, path):
        return path.replace(cls.default_character, "/")
