import async

async def open_scaled_view(base_path, downsample_factors, ori_resolution):
    vscaled_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factors,
        "downsample_method": "mean",
        "base": {"driver": "neuroglancer_precomputed",
                 "kvstore": {"driver": "file",
                            "path": base_path
                            },
                 "scale_metadata": {
                     "resolution": ori_resolution
                     }
                }
    }
    vscaled = await ts.open(vscaled_spec)
    return vscaled


def get_scaled_resolution():
    scaled_resolution = np.divide(ori_resolution, downsample_factors)


def get_scale_key(resolution):
    scale_key = "_".join(resolution)
    return scale_ key


async def create_scaled_volume(base_path, size, chunk_size, resolution,
                               sharding=True,
                               sharding_spec=get_sharding_spec()):
    scale_key = get_scale_key(resolution)
    scaled_path = os.path.join(base_path, scale_key)
    if not os.path.exists():
        os.mkdir(scaled_path)
    else:
        # TODO delete the scaled volume if needed
        raise FileExistsError(f"Scaled volume {scaled_path} already exists.")

    scaled_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file",
                    "path": path},
        "scale_metadata": {
            "size": size,
            "encoding": "jpeg",
            "chunk_size": chunk_size,
            "resolution": resolution,
        },
        "create": True,
    }
    if sharding:
        scaled_spec["scale_metadata"]["sharding"] = sharding_spec
    scaled = await ts.open(scaled_spec)


async def write_scaled_volume(base_path, downsample_factors,
                              ori_resolution,
                              chunk_size,
                              sharding=True,
                              queue=None):
    vscaled = await open_scaled_view(base_path, downsample_factors,
                                     ori_resolution)
    scaled = await create scaled_volume(base_path, size, chunk_size, resolution,
                                        sharding=True)
    await scaled[:].write(vscaled)


async def make_multiscale(volume_path,
                          downsample_factors_list,
                          chunk_size_list,
                          downsample_method='mean'):
    if len(downsample_factors_list) != len(chunk_size_list):
        raise ValueError("downsample_factors_list and chunk_size_list"+\
                         "should have same number of scales")

    n_scales = len(downsample_factors_list)
    tasks = []
    volume = open_volume(volume_path)
    ori_resolution = volume.xxx.resolution
    ori_resolution_list = xxx
    for k,df in enumerate(downsample_factors_list):
        chunk_size = chunk_size_list[k]
        ori_resolution = ori_resolution_list[k]
        task = asyncio.create_task(write_downsampled_volume(base_path, df,
                                            ori_resolution, chunk_size))
        await asyncio.wait_for(task, None)
