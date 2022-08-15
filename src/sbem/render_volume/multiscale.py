import os
import asyncio
import numpy as np
import tensorstore as ts

from sbem.render_volume.render_utils import (
    get_scale_key, get_sharding_spec,
    open_volume, get_resolution,
    get_shard_box, box_to_slices)
from sbem.render_volume.schema import SizeHiearchy


async def open_scaled_view(base_path, base_scale_key, downsample_factors,
                           downsample_method):
    vscaled_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factors,
        "downsample_method": downsample_method,
        "base": {"driver": "neuroglancer_precomputed",
                 "kvstore": {"driver": "file",
                            "path": base_path
                            },
                 "scale_metadata": {
                      "key": base_scale_key
                     }
                }
    }
    vscaled = await ts.open(vscaled_spec)
    return vscaled


def get_scaled_resolution():
    scaled_resolution = np.divide(ori_resolution, downsample_factors)


async def create_scaled_volume(base_path, scale_key,
                               size, chunk_size, resolution,
                               sharding=True,
                               sharding_spec=get_sharding_spec()):
    scale_key = get_scale_key(resolution)
    scaled_path = os.path.join(base_path, scale_key)

    scaled_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file",
                    "path": base_path},
        "scale_metadata": {
            "size": size,
            "encoding": "raw",
            "chunk_size": chunk_size,
            "resolution": resolution,
            "key": scale_key
        },
    }

    if not os.path.exists(scaled_path):
        os.mkdir(scaled_path)
        scaled_spec["create"] = True

    if sharding:
        scaled_spec["scale_metadata"]["sharding"] = sharding_spec
    scaled = await ts.open(scaled_spec)
    return scaled


async def write_scaled_volume(base_path,
                              base_scale_key,
                              scale_key,
                              downsample_factors,
                              chunk_size,
                              downsample_method,
                              sharding=True,
                              shard_bits_list=(9,6,15)):
    vscaled = await open_scaled_view(base_path, base_scale_key,
                                     downsample_factors,
                                     downsample_method)

    volume_size = vscaled.shape[:-1]
    resolution = get_resolution(vscaled)
    sharding_spec = get_sharding_spec(*shard_bits_list)
    scaled = await create_scaled_volume(base_path, scale_key,
                                        volume_size, chunk_size, resolution,
                                        sharding=True,
                                        sharding_spec=sharding_spec)

    preshift_bits, minishard_bits, shard_bits = shard_bits_list

    hierarchy = SizeHiearchy(volume_size=volume_size,
                             chunk_size=chunk_size)

    hierarchy.shard_size_in_chunks = (preshift_bits+minishard_bits)/3
    hierarchy.shard_size = np.multiply(hierarchy.shard_size_in_chunks,
                                           hierarchy.chunk_size).astype(int)
    hierarchy.grid_shape_in_shards = np.ceil(
        np.divide(hierarchy.volume_size, hierarchy.shard_size)).astype(int)

    num_shards = hierarchy.grid_shape_in_shards

    for i in range(num_shards[0]):
        for j in range(num_shards[1]):
            for k in range(num_shards[2]):
                shard_index_xyz = (i, j, k)
                print(shard_index_xyz)
                box = get_shard_box(shard_index_xyz, hierarchy.shard_size,
                                    volume_size)
                slices = box_to_slices(box)
                await scaled[slices[0], slices[1], slices[2], 0].write(
                    vscaled[slices[0], slices[1], slices[2]])


async def make_multiscale(volume_path,
                          downsample_factors_list,
                          chunk_size_list,
                          sharding_list,
                          base_scale_key=None,
                          downsample_method='stride'):
    if len(downsample_factors_list) != len(chunk_size_list):
        raise ValueError("downsample_factors_list and chunk_size_list"+\
                         "should have same number of scales")

    n_scales = len(downsample_factors_list)
    tasks = []
    if base_scale_key is None:
        volume = await open_volume(volume_path, scale_index=0)
    else:
        volume = await open_volume(volume_path, scale_key=base_scale_key)

    base_resolution = get_resolution(volume)
    volume_scale_key = get_scale_key(base_resolution)
    resolutions = []
    for df in downsample_factors_list:
        base_resolution = np.multiply(base_resolution, df[:-1]).astype(int)
        resolutions.append(base_resolution)

    scale_keys = [get_scale_key(r) for r in resolutions]
    scale_keys.insert(0, volume_scale_key)

    for k,df in enumerate(downsample_factors_list):
        chunk_size = chunk_size_list[k]
        base_scale_key = scale_keys[k]
        scale_key = scale_keys[k+1]
        print(scale_key)
        task = asyncio.create_task(
            write_scaled_volume(volume_path,
                                base_scale_key,
                                scale_key,
                                df,
                                chunk_size,
                                downsample_method,
                                sharding=True,
                                shard_bits_list=sharding_list[k]))
        await asyncio.wait_for(task, None)
