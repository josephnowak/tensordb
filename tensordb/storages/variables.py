from tensordb.storages.zarr_storage import ZarrStorage
from tensordb.storages.json_storage import JsonStorage

MAPPING_STORAGES = {
    'zarr_storage': ZarrStorage,
    'json_storage': JsonStorage
}
