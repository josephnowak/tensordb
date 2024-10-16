from tensordb.storages.json_storage import JsonStorage
from tensordb.storages.zarr_storage import ZarrStorage

MAPPING_STORAGES = {"zarr_storage": ZarrStorage, "json_storage": JsonStorage}
