import xarray
import orjson
import numpy as np
import fsspec

from typing import Dict
from pandas import Timestamp
from zarr.util import NoLock
from loguru import logger


no_lock = NoLock()


def get_lock(synchronizer, path):
    if synchronizer is None:
        return no_lock
    return synchronizer[path]


def get_dims(path_map: fsspec.FSMap, name: str):
    return orjson.loads(path_map[f'{name}/.zattrs'])['_ARRAY_DIMENSIONS']


def get_chunks_sizes(path_map: fsspec.FSMap, name: str):
    return orjson.loads(path_map[f'{name}/.zarray'])['chunks']


def find_positions(x: np.array, y: np.array):
    sorted_keys = np.argsort(x)
    return sorted_keys[np.searchsorted(x, y, sorter=sorted_keys)]


def get_affected_chunks(path_map: fsspec.FSMap, actual_coords, coords, data_name):
    dims = get_dims(path_map, data_name)
    chunks_positions_data = []
    chunks_names = []
    chunks_sizes_data = get_chunks_sizes(path_map, data_name)
    for i, dim in enumerate(dims):
        # validate the dtype of the coord to modify
        coord = coords[dim].values if isinstance(coords[dim], xarray.DataArray) else coords[dim]

        # Find the positions that were affected
        positions = find_positions(actual_coords[dim].values, coord)

        # Apply the zarr logic to find the chunk names affected in the data
        chunks_positions_data.append(np.unique(positions // chunks_sizes_data[i]))

        #  Apply the zarr logic to find the chunk names affected in the coord
        chunks_dim = get_chunks_sizes(path_map, dim)
        chunks_names += [f'{dim}/{chunk_name}' for chunk_name in np.unique(positions // chunks_dim[0])]

        # By default add the zattrs and zarray of the coord
        chunks_names += [f'{dim}/.zattrs', f'{dim}/.zarray']

    # By default add the zattrs and zarray of the data
    chunks_names += [f'{data_name}/.zattrs', f'{data_name}/.zarray']

    chunks_names += [
        data_name + '/' + ".".join(map(str, block))
        for block in np.array(np.meshgrid(*chunks_positions_data)).T.reshape(-1, len(chunks_positions_data))
    ]

    return chunks_names


def update_checksums_temp(local_map: fsspec.FSMap, chunks_name):
    date = str(Timestamp.now())
    checksums = {chunk_name: str(local_map.fs.checksum(f'{local_map.root}/{chunk_name}')) for chunk_name in chunks_name}
    local_map['temp_checksums.json'] = orjson.dumps(checksums)
    local_map['temp_last_modification_date.json'] = orjson.dumps({'date': date})


def update_checksums(path_map: fsspec.FSMap, chunks_name):
    date = str(Timestamp.now())
    checksums = {chunk_name: str(path_map.fs.checksum(f'{path_map.root}/{chunk_name}')) for chunk_name in chunks_name}
    total_checksums = orjson.loads(path_map['checksums.json'])
    total_checksums.update(checksums)
    path_map['checksums.json'] = orjson.dumps(total_checksums)
    path_map['last_modification_date.json'] = orjson.dumps({'last_modification_date': date})


def merge_local_checksums(local_map: fsspec.FSMap):
    if 'temp_checksums.json' not in local_map:
        return

    checksums = {}
    if 'checksums.json' in local_map:
        checksums = orjson.loads(local_map['checksums.json'])
    checksums.update(orjson.loads(local_map['temp_checksums.json']))

    local_map['checksums.json'] = orjson.dumps(checksums)
    local_map['last_modification_date.json'] = local_map['temp_last_modification_date.json']

    del local_map['temp_checksums.json']
    del local_map['temp_last_modification_date.json']

