import xarray
import orjson

from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Generic

from tensordb.file_handlers.base_handler import BaseStorage


class JsonStorage(BaseStorage):

    def store(self, name: str, new_data: Dict, **kwargs):
        new_name = name.replace('\\', '/').replace('/', '_')
        self.backup_map[new_name] = orjson.dumps(new_data)

    def append(self, name: str, new_data: Dict, **kwargs):
        raise NotImplemented('Use upsert')

    def update(self, name: str, new_data: Dict, **kwargs):
        raise NotImplemented('Use upsert')

    def upsert(self, name: str, new_data: Dict, **kwargs):
        new_name = name.replace('\\', '/').replace('/', '_')
        if new_name not in self.backup_map:
            raise KeyError(f'The file with name {name} does not exist, so you can not upsert data. '
                           f'Use the store method')

        act_data = orjson.loads(self.backup_map[new_name])
        act_data.update(new_data)
        self.backup_map[new_name] = orjson.dumps(act_data)

    def read(self, name: str, **kwargs) -> xarray.DataArray:
        new_name = name.replace('\\', '/').replace('/', '_')
        return orjson.loads(self.backup_map[new_name])

    def set_attrs(self, **kwargs) -> xarray.DataArray:
        raise NotImplemented('Use the append method inplace of set_attrs')

    def get_attrs(self, **kwargs) -> xarray.DataArray:
        raise NotImplemented('Use the read method inplace of get_attrs')

    def update_from_backup(self, **kwargs):
        raise NotImplemented('All is stored in the backup, you do not need to update from backup')

    def backup(self, **kwargs):
        raise NotImplemented('All is stored in the backup, you do not need to manually call backup method')

    def close(self, **kwargs):
        raise NotImplemented('The files are not keep open')

    def exist(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', '_')
        return new_name in self.backup_map

    def delete_file(self, name: str, **kwargs):
        new_name = name.replace('\\', '/').replace('/', '_')
        del self.backup_map[new_name]




