from tensordb.backends.zarr import ZarrBackend
from tensordb.backends.json import JsonBackend
from tensordb.backends.base import BaseBackend
from typing import Dict


ENGINES: Dict[str, BaseBackend] = {
    "zarr": ZarrBackend,
    "json": JsonBackend
}
