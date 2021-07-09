from tensordb.file_handlers import BaseStorage, JsonStorage, ZarrStorage

MAPPING_STORAGES = {
    'zarr_storage': ZarrStorage,
    'json_storage': JsonStorage,
    'base_storage': BaseStorage
}
