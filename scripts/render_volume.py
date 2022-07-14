import os
import logging
import zarr
import ome_zarr
import numpy as np
from numcodecs import Blosc
from sbem.section_align.align_utils import (
    load_sections, get_section_pairs, get_pair_name, load_json)
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_multiscale
from pybdv import make_bdv_from_dask_array
from dask.array import from_zarr
from tqdm import tqdm


def load_offsets(offset_dir, sbem_experiment, grid_index, start_section, end_section):
    logger = logging.getLogger()
    sections = load_sections(sbem_experiment, grid_index, start_section, end_section, logger=logger)
    section_pairs = get_section_pairs(sections)

    xyo_list = []
    for pair in section_pairs:
        pair_name = get_pair_name(pair)
        offset_path = os.path.join(offset_dir, f"{pair_name}.json")
        try:
            offset = load_json(offset_path)
        except FileNotFoundError as e:
            print("No offset file found")
            raise e

        if offset == "error":
            msg = f"Alignment for {pair_name} had an error"
            raise ValueError(msg)

        xyo = offset["xyo"]
        xyo_list.append(xyo)

    xy_offsets = np.array(xyo_list)
    return xy_offsets, sections


def solve_offsets_to_coords(xy_offsets):
    xy_coords = np.cumsum(xy_offsets, axis=0)
    return xy_coords


def render_volume(volume_path, sections, xy_coords):
    logger = logging.getLogger()

    n_sections = len(sections)
    height = 28160
    width = 40192
    zarr_volume = zarr.open(volume_path, mode="w", shape=(n_sections, height, width),
                chunks=(1, 4000, 4000), dtype=np.uint8,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE))
    zarr_section = []
    for k, section in tqdm(enumerate(sections), "Writing sections"):
        # Open stitched image
        stitched = section.read_stitched()
        zarr_section.append(stitched)
        zarr_volume[k,:stitched.shape[0],:stitched.shape[1]] = zarr_section[k]
    return zarr_volume


def convert_to_multiscale(zarr_volume, file_path):
    # create multiscale image pyramid (mip) using the scaler class
    print("scaling")
    scaler = Scaler(downscale=32, in_place=False, max_layer=2)
    mip = scaler.local_mean(zarr_volume)
    print("scaling done")
    storage_options=dict(chunks=(1, size_xy, size_xy))
    # create a zarr handle in write mode
    loc = ome_zarr.io.parse_url(file_path, mode="w")

    # create a zarr root level group at the file path
    group = zarr.group(loc.store)

    # write the actual data
    write_multiscale(mip, group, axes=("z", "x", "y"),
                     storage_options=storage_options)

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

if __name__ == "__main__":
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
    xy_coords = solve_offsets_to_coords(xy_offsets)

    render_volume(volume_path, sections, xy_coords)
    zarr_volume = zarr.open(volume_path, "r")
    convert_to_multiscale_n5(zarr_volume, multiscale_path)
