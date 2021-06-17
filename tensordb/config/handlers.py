from tensordb.file_handlers import BaseStorage, JsonStorage, ZarrStorage

mapping_storages = {
    'zarr_storage': ZarrStorage,
    'json_storage': JsonStorage,
    'base_storage': BaseStorage
}
