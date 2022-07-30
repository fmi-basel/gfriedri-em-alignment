import os
import asyncio
import logging
import json
from tqdm import tqdm
import numpy as np
import tensorstore as ts

from typing import List
from sbem.record import SectionRecord

import tracemalloc

async def read_n5(path: str):
    """
    Read N5 dataset

    :param path: path to the N5 dataset

    :return: `tensorstore.TensorStore`
    """
    dataset_future = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': path,
            },
        "context": {
            "cache_pool": {"total_bytes_limit": 10*1024*1024},
            "data_copy_concurrency": {"limit": 6},
            'file_io_concurrency': {'limit': 6}
            }
        },read=True)
    dataset = await dataset_future
    return dataset


async def read_stitched(section: SectionRecord):
    """
    Read stitched image of a section

    :param section: `SectionRecord`

    :return: `tensorstore.TensorStore`
    """

    stitched_path = section.get_stitched_path()
    stitched = await read_n5(stitched_path)
    return stitched


async def read_stitched_sections(sections: List["SectionRecord"]):
    stitched_sections = []

    for section in sections:
        stitched = await read_stitched(section)
        stitched_sections.append(stitched)

    return stitched_sections


def get_sharding_spec(shard_bits=15):
    sharding_spec =  {
        "@type": "neuroglancer_uint64_sharded_v1",
        "data_encoding": "gzip",
        "hash": "identity",
        "minishard_bits": 6,
        "minishard_index_encoding": "gzip",
        "preshift_bits": 9,
        "shard_bits": shard_bits
        }
    return sharding_spec


async def create_volume(path: str,
                        size: list,
                        chunk_size: list,
                        resolution: list,
                        sharding: bool=True,
                        sharding_spec: dict=get_sharding_spec()):
    """
    Create a multiscale neuroglancer_precomputed volume
    with a specified single scale

    :param path
    :param size
    :param chunk_size
    :param resolution
    :param sharding
    """
    volume_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file",
                    "path": path
                    },
        "multiscale_metadata": {
            "type": "image",
            "data_type": "uint8",
            "num_channels": 1
            },
        "scale_metadata": {
            "size": size,
            "encoding": "raw",
            "chunk_size": chunk_size,
            "resolution": resolution,
            },
        "context": {
            "cache_pool": {"total_bytes_limit": 20*1024*1024*1024},
            "data_copy_concurrency": {"limit": 6}
            },
        "create": True,
        "delete_existing": True
        }

    if sharding:
        volume_spec["scale_metadata"]["sharding"] = sharding_spec

    volume_future = ts.open(volume_spec)

    volume = await volume_future
    return volume


async def open_volume(path, scale_index=0, scale_key=None):
    volume_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file",
                    "path": path
                    },
        }
    if scale_key:
        volume_spec["scale_metadata"] = dict({"key": scale_key})
    else:
        volume_spec["scale_index"] = scale_index

    volume = await ts.open(volume_spec)
    return volume


def get_scale_key(resolution):
    # This assumes resolution is integer
    scale_key = "_".join(map(lambda x: str(int(x)), resolution))
    return scale_key


def get_resolution(volume):
    # this assumes that each dimension unit is of the form [number, string]
    # and resolution is integer
    dimension_units = volume.dimension_units
    resolution = [int(du.multiplier) for du in dimension_units[:-1]]
    return resolution


async def estimate_volume_size(stitched_sections, xy_coords):
    n_sections = len(stitched_sections)
    max_xy = xy_coords.max(axis=0)
    shape_list = np.array([s.shape for s in stitched_sections])
    shape_max = shape_list.max(axis=0)
    width = shape_max[0] + max_xy[0]
    height = shape_max[1] + max_xy[1]
    volume_size = list(map(int, [width, height, n_sections]))
    return volume_size


async def render_volume(volume_path: str,
                        sections: List["SectionRecord"],
                        xy_coords: np.ndarray,
                        resolution: List[int],
                        chunk_size: List[int]=[64, 64, 64]):
    """
    Render a range of sections into a 3d wolume

    :param volume_path: the path for writing the volume.
    :param sections: a list of SectionRecord objects
    :param xy_coords: N*2 array, N equals number of sections.
                      Each row is XY offset of each section, with respect
                      to the first secion.
    :param resolution: resolution in nanometer in X, Y, Z.
    :param chunk_size: Chunk size for saving chunked volume. Each element
                       corresponds to dimension X, Y, Z.

    :return volume: `tensorstore.TensorStore` referring to the created volume
    """
    if np.any(xy_coords < 0):
        raise ValueError("The XY offset (xy_coords) should be non-negative.")
    tracemalloc.start()



    stitched_sections = await read_stitched_sections(sections)
    volume_size = await estimate_volume_size(stitched_sections, xy_coords)
    sharding_spec = get_sharding_spec(shard_bits=4)
    volume = await create_volume(volume_path, volume_size, chunk_size,
                                 resolution,
                                 sharding=True, sharding_spec=sharding_spec)
    print(f"volume_size: {volume_size}")

    shard_size = 64*5
    nshard_x = np.ceil(volume_size[0] / shard_size).astype(int)
    nshard_y = np.ceil(volume_size[1] / shard_size).astype(int)
    print(f"nshard {nshard_x}, {nshard_y}")
    nshard_x = 60
    nshard_y = 60
    print(f"nshard {nshard_x}, {nshard_y}")

    for i in range(nshard_x):
        xs = slice(i*shard_size,min((i+1)*shard_size, volume_size[0]))
        for j in range(nshard_y):
            ys = slice(j*shard_size,min((j+1)*shard_size, volume_size[1]))
            print(i,j)
            txn = ts.Transaction()
            for k, section in enumerate(sections):
                #TODO xyo
                xyo = xy_coords[k]
                stitched = stitched_sections[k][xs, ys]
                await volume[xs, ys, k, 0].with_transaction(txn).write(stitched)
            print("Start writing")
            await txn.commit_async()

    return volume
# sum_i ceil(log_2(shape[i] / chunk_shape[i]))
