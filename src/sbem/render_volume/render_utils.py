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
            "cache_pool": {"total_bytes_limit": 40*1024*1024},
            "data_copy_concurrency": {"limit": 8},
            'file_io_concurrency': {'limit': 8}
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


def downsample_sections(stitched_sections,
                        downsample_factors,
                        method):
    dsampled_sections = []

    for ssec in stitched_sections:
        ds = ts.downsample(ssec, downsample_factors, method)
        dsampled_sections.append(ds)

    return dsampled_sections


def get_sharding_spec(preshift_bits=9, minishard_bits=6, shard_bits=15):
    sharding_spec =  {
        "@type": "neuroglancer_uint64_sharded_v1",
        "data_encoding": "gzip",
        "hash": "identity",
        "minishard_bits": minishard_bits,
        "minishard_index_encoding": "gzip",
        "preshift_bits": preshift_bits,
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
            "data_copy_concurrency": {"limit": 20},
            'file_io_concurrency': {'limit': 20}
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


def pick_shard_bits(volume_size, chunk_size,
                    preshift_bits, minishard_bits):
    grid_shape_in_chunks = np.ceil(np.divide(volume_size, chunk_size))
    bits = np.ceil(np.log2(np.maximum(0, grid_shape_in_chunks - 1)))
    total_z_index_bits = np.sum(bits)
    shard_bits = total_z_index_bits - (preshift_bits + minishard_bits)
    shard_bits = int(shard_bits)
    if shard_bits <= 0:
        msg = f"non_shard_bits {preshift_bits}+{minishard_bits}"+\
              f"should be less than total_z_index_bits {total_z_index_bits}"
        raise ValueError(msg)
    return shard_bits, bits

def prepare_render_volume():
    pass


def write_volume():
    pass



async def render_volume(volume_path: str,
                        sections: List["SectionRecord"],
                        xy_coords: np.ndarray,
                        resolution: List[int],
                        chunk_size: List[int]=[64, 64, 64],
                        downsample: bool=False,
                        downsample_factors: List[int]=[1, 1],
                        downsample_method: str="mean",
                        preshift_bits=9,
                        minishard_bits=6):
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
    if downsample:
        stitched_sections = downsample_sections(stitched_sections,
                                                downsample_factors,
                                                downsample_method)
        xy_coords = np.ceil(np.divide(xy_coords,downsample_factors)).astype(int)

    volume_size = await estimate_volume_size(stitched_sections, xy_coords)


    shard_bits, bits_xyz = pick_shard_bits(volume_size, chunk_size,
                                    preshift_bits, minishard_bits)
    print(f"bits_xyz: {bits_xyz}")

    sharding_spec = get_sharding_spec(preshift_bits=preshift_bits,
                                      minishard_bits=minishard_bits,
                                      shard_bits=shard_bits)

    volume = await create_volume(volume_path, volume_size, chunk_size,
                                 resolution,
                                 sharding=True, sharding_spec=sharding_spec)
    print(f"volume_size: {volume_size}")

    # This estimation of shard_size requires x,y,z
    # non-zero bits all more than (preshift_bits+minishard_bits)/3
    shard_size = np.multiply((preshift_bits+minishard_bits)/3,
                             chunk_size).astype(int)

    num_shards = np.ceil(np.divide(volume_size, shard_size)).astype(int)
    print(f"num_shards: {num_shards}")

    # The order of dimensions is XYZ
    for k in range(num_shards[2]):
        for i in range(num_shards[0]):
            for j in range(num_shards[1]):
                shard_index_xyz = (i, j, k)
                box = _get_shard_box(shard_index_xyz, shard_size,
                                    volume_size)
                txn = ts.Transaction()
                for z in range(*box[2]):
                    stitched = stitched_sections[z]
                    xyo = xy_coords[z]
                    box_xy = _get_shifted_box(box[:2], xyo)
                    box_xy = _limit_box_by_total_size(box_xy, stitched.shape)
                    slices_xy = _box_to_slices(box_xy)
                    target_box_xy = _limit_box_by_another_box_size(box[:2],
                                                                   box_xy)
                    target_slices = _box_to_slices(target_box_xy)

                    source = stitched[slices_xy[0], slices_xy[1]]
                    await volume[target_slices[0], target_slices[1],
                                 z, 0].with_transaction(txn).write(source)
                print(f"Start writing {shard_index_xyz}")
                await txn.commit_async()

        for z in range(*box[2]):
            stitched_sections[z] = None

    return volume


def _get_shard_box(shard_index_xyz, shard_size, volume_size):
    box = [[int(i*s), int((i+1)*s)] for i,s in zip(shard_index_xyz, shard_size)]
    box = _limit_box_by_total_size(box, volume_size)
    return box


def _get_shifted_box(box_xy, xy_offset):
    shifted_box_xy = np.array(np.maximum(0, box_xy-np.tile(xy_offset,(2,1)).T),
                              dtype=int)
    return shifted_box_xy


def _limit_box_by_total_size(box, total_size):
    limited_box = [[max(0, min(b[0], s)),
                    max(0, min(b[1], s))]
            for b, s in zip(box, total_size)]
    return limited_box


def _box_to_slices(box):
    slices = [slice(b[0], b[1])for b in box]
    return slices


def _limit_box_by_another_box_size(box, limit_box):
    lbox = [[b[0], min(b[1], b[0]+lb[1]-lb[0])]
            for b, lb in zip(box, limit_box)]
    return lbox
