import os
import logging
import asyncio
import zarr
import numpy as np
from numcodecs import Blosc
from sbem.section_align.align_utils import (load_offsets, offsets_to_coords)
from pybdv import make_bdv_from_dask_array
from dask.array import from_zarr
from tqdm import tqdm


async def read_stitched(section):
    pass


async def render_volume(volume_path, sections, xy_coords):
    logger = logging.getLogger()

    n_sections = len(sections)
    height = 28160
    width = 40192
    await zarr_volume = zarr.open(volume_path, mode="w", shape=(n_sections, height, width),
                chunks=(1, 4000, 4000), dtype=np.uint8,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE))
    zarr_section = []
    for k, section in tqdm(enumerate(sections), "Writing sections"):
        # Open stitched image
        await stitched = read_stitched(section)
        zarr_section.append(stitched)
        zarr_volume[k,:stitched.shape[0],:stitched.shape[1]] = zarr_section[k]
    return zarr_volume


def convert_to_multiscale_n5(zarr_volume, file_path):
    scale_factors = [[2, 2, 2], [2, 2, 2], [4, 4, 4]]
    mode = 'mean'
# specify a resolution of 0.5 micron per pixel (for zeroth scale level)
    dask_arr_volume = from_zarr(zarr_volume)
    make_bdv_from_dask_array(dask_arr_volume, file_path,
                downscale_factors=scale_factors, downscale_mode=mode,
                resolution=[33, 11, 11], unit='nanometer',
                chunks=(256,256,256),
                downsample_chunks=[(128,128,128), (64,64,64), (32,32,32)]
    )


async def main():
    sbem_experiment="path/to/experiment"
    grid_index=1
    start_section=5000
    end_section=5010
    volume_name = f"s{start_section}_s{end_section}"
    volume_path = os.path.join(sbem_experiment, "volume", volume_name+".zarr")
    multiscale_path = os.path.join(sbem_experiment, "volume", "ms_"+volume_name+".n5")

    downscale_factor = 8


    offset_dir = os.path.join(sbem_experiment, "zalign")

    xy_offsets, sections = load_offsets(offset_dir, sbem_experiment, grid_index, start_section, end_section)
    xy_coords = offsets_to_coords(xy_offsets)

    await volume = render_volume(volume_path, sections, xy_coords)
    # zarr_volume = zarr.open(volume_path, "r")
    await convert_to_multiscale_n5(volume, multiscale_path)


if __name__ == "__main__":
    asyncio.run(main())
