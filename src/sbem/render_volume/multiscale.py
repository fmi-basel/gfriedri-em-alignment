import os
import asyncio
import numpy as np
import tensorstore as ts

from sbem.render_volume.render_utils import (
    get_scale_key, get_sharding_spec,
    open_volume, get_resolution)

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
            "encoding": "jpeg",
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
                              queue=None):
    vscaled = await open_scaled_view(base_path, base_scale_key,
                                     downsample_factors,
                                     downsample_method)

    size = vscaled.shape[:-1]
    resolution = get_resolution(vscaled)
    scaled = await create_scaled_volume(base_path, scale_key,
                                        size, chunk_size, resolution,
                                        sharding=True)
    await scaled[:].write(vscaled)


async def make_multiscale(volume_path,
                          downsample_factors_list,
                          chunk_size_list,
                          downsample_method='stride'):
    if len(downsample_factors_list) != len(chunk_size_list):
        raise ValueError("downsample_factors_list and chunk_size_list"+\
                         "should have same number of scales")

    n_scales = len(downsample_factors_list)
    tasks = []
    volume = await open_volume(volume_path, scale_index=0)

    base_resolution = get_resolution(volume)
    volume_scale_key = get_scale_key(base_resolution)
    resolutions = [np.multiply(base_resolution, df[:-1]).astype(int)
                   for df in downsample_factors_list]
    scale_keys = [get_scale_key(r) for r in resolutions]
    scale_keys.insert(0, volume_scale_key)

    for k,df in enumerate(downsample_factors_list):
        chunk_size = chunk_size_list[k]
        base_scale_key = scale_keys[k]
        scale_key = scale_keys[k+1]
        task = asyncio.create_task(
            write_scaled_volume(volume_path,
                                base_scale_key,
                                scale_key,
                                df,
                                chunk_size,
                                downsample_method))
        await asyncio.wait_for(task, None)
