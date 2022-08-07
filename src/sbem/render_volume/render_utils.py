import os
import asyncio
import logging
import json
from tqdm import tqdm
import numpy as np
import tensorstore as ts

from typing import List
from sbem.record import SectionRecord
from sbem.render_volume.schema import SizeHiearchy


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


def pick_shard_bits(bits_xyz,
                    preshift_bits, minishard_bits):
    total_z_index_bits = np.sum(bits_xyz)
    shard_bits = total_z_index_bits - (preshift_bits + minishard_bits)
    shard_bits = int(shard_bits)
    if shard_bits <= 0:
        msg = f"non_shard_bits {preshift_bits}+{minishard_bits}"+\
              f" should be less than total_z_index_bits {total_z_index_bits}"
        raise ValueError(msg)
    return shard_bits


async def load_stitched_and_prepare_volume(sections, xy_coords,
                                           chunk_size, resolution,
                                           downsample_config=None):
    if np.any(xy_coords < 0):
        raise ValueError("The XY offset (xy_coords) should be non-negative.")


    stitched_sections = await read_stitched_sections(sections)
    if downsample_config is not None:
        stitched_sections = downsample_sections(stitched_sections,
                                                **downsample_config.to_dict())
        dfs = downsample_config.downsample_factors
        xy_coords = np.ceil(np.divide(xy_coords, dfs)).astype(int)
        resolution[:2] = np.multiply(resolution[:2], dfs)

    volume_size = await estimate_volume_size(stitched_sections, xy_coords)

    grid_shape_in_chunks = np.ceil(np.divide(volume_size, chunk_size))

    bits_xyz =  np.ceil(np.log2(np.maximum(0, grid_shape_in_chunks - 1)))

    size_hierarchy = SizeHiearchy(volume_size=volume_size,
                                  chunk_size=chunk_size,
                                  grid_shape_in_chunks=grid_shape_in_chunks,
                                  bits_xyz=bits_xyz)

    return stitched_sections, xy_coords, size_hierarchy, resolution


def prepare_sharding(hierarchy, preshift_bits, minishard_bits):
    shard_bits = pick_shard_bits(hierarchy.bits_xyz,
                                 preshift_bits, minishard_bits)

    # This estimation of shard_size requires x,y,z
    # non-zero bits all more than (preshift_bits+minishard_bits)/3
    hierarchy.shard_size_in_chunks = (preshift_bits+minishard_bits)/3
    hierarchy.shard_size = np.multiply(hierarchy.shard_size_in_chunks,
                                           hierarchy.chunk_size).astype(int)
    hierarchy.grid_shape_in_shards = np.ceil(
        np.divide(hierarchy.volume_size, hierarchy.shard_size)).astype(int)

    sharding_spec = get_sharding_spec(preshift_bits=preshift_bits,
                                      minishard_bits=minishard_bits,
                                      shard_bits=shard_bits)


    return sharding_spec, hierarchy


async def write_volume(volume, stitched_sections, xy_coords, size_hierarchy):
    # (i, j, k) corresponds to XYZ
    num_shards = size_hierarchy.grid_shape_in_shards
    for k in range(num_shards[2]):
        for i in range(num_shards[0]):
            for j in range(num_shards[1]):
                shard_index_xyz = (i, j, k)
                box = _get_shard_box(shard_index_xyz, size_hierarchy.shard_size,
                                    size_hierarchy.volume_size)
                txn = ts.Transaction()
                for z in range(*box[2]):
                    stitched = stitched_sections[z]
                    xyo = xy_coords[z]
                    box_xy = _get_shifted_box(box[:2], xyo)

                    source_box_xy = _limit_box_by_total_size(box_xy,
                                                             stitched.shape)
                    source_slices_xy = _box_to_slices(source_box_xy)

                    target_box_xy = _get_shifted_box(source_box_xy, -xyo)
                    target_slices = _box_to_slices(target_box_xy)

                    source = stitched[source_slices_xy[0], source_slices_xy[1]]

                    await volume[target_slices[0], target_slices[1],
                                 z, 0].with_transaction(txn).write(source)
                print(f"Start writing {shard_index_xyz}")
                await txn.commit_async()

        for z in range(*box[2]):
            stitched_sections[z] = None





def _get_shard_box(shard_index_xyz, shard_size, volume_size):
    box = [[int(i*s), int((i+1)*s)] for i,s in zip(shard_index_xyz, shard_size)]
    box = _limit_box_by_total_size(box, volume_size)
    return box


def _get_shifted_box(box_xy, xy_offset):
    shifted_box_xy = np.array(box_xy-np.tile(xy_offset,(2,1)).T,
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


# def _limit_box_by_another_box(box, limit_box):
#     lbox = [[xxxx
#              min(b[1], b[0]+lb[1]-lb[0])]
#             for b, lb in zip(box, limit_box)]
#     return lbox
