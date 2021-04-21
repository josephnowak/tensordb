import xarray
import json
import numpy as np
import zarr
import fsspec

from typing import Dict
from pandas import Timestamp


def get_dims(path_map: fsspec.FSMap, name: str):
    return json.loads(path_map[f'{name}/.zattrs'])['_ARRAY_DIMENSIONS']


def get_chunks_sizes(path_map: fsspec.FSMap, name: str):
    return json.loads(path_map[f'{name}/.zarray'])['chunks']


def find_positions(x: np.array, y: np.array):
    sorted_keys = np.argsort(x)
    return sorted_keys[np.searchsorted(x, y, sorter=sorted_keys)]


def get_affected_chunks(path_map: fsspec.FSMap, coords, data_name):
    dims = get_dims(path_map, data_name)
    chunks_positions_data = []
    chunks_names = []
    chunks_sizes_data = get_chunks_sizes(path_map, data_name)
    for i, dim in enumerate(dims):
        if dim not in coords:
            continue
        # oppening the coord as a zarr array
        arr = zarr.open(fsspec.FSMap(f'{path_map.root}/{dim}', path_map.fs), mode='r')

        # validate the dtype of the coord to modify
        coord = coords[dim].values if isinstance(coords[dim], xarray.DataArray) else coords[dim]

        # Find the positions that were affected
        positions = find_positions(arr[:], coord)

        # Apply the zarr logic to find the chunk names affected in the data
        chunks_positions_data.append(np.unique(positions // chunks_sizes_data[i]))

        #  Apply the zarr logic to find the chunk names affected in the coord
        chunks_names += [f'{dim}/{chunk_name}' for chunk_name in np.unique(positions // arr.chunks[0])]

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
    local_map['temp_checksums.json'] = json.dumps(checksums).encode('utf-8')
    local_map['temp_last_modification_date.json'] = json.dumps({'date': date}).encode('utf-8')


def update_checksums(path_map: fsspec.FSMap, chunks_name):
    date = str(Timestamp.now())
    checksums = {chunk_name: str(path_map.fs.checksum(f'{path_map.root}/{chunk_name}')) for chunk_name in chunks_name}
    total_checksums = json.loads(path_map['checksums.json'])
    total_checksums.update(checksums)
    path_map['checksums.json'] = json.dumps(modified_chunks).encode('utf-8')
    path_map['last_modification_date.json'] = json.dumps({'last_modification_date': date}).encode('utf-8')


def merge_local_checksums(local_map: fsspec.FSMap):
    if not local_map.fs.exists(f'{local_map.root}/temp_checksums.json'):
        return

    checksums = {}
    if local_map.fs.exists(f'{local_map.root}/checksums.json'):
        checksums = json.loads(local_map['checksums.json'])
    checksums.update(json.loads(local_map['temp_checksums.json']))

    local_map['checksums.json'] = json.dumps(checksums).encode('utf-8')
    local_map.fs.rm(f'{local_map.root}/temp_checksums.json')
    local_map['last_modification_date.json'] = local_map['temp_last_modification_date.json']
    local_map.fs.rm(f'{local_map.root}/temp_last_modification_date.json')

