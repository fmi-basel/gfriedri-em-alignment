import os
import asyncio
import logging
import json

from sbem.section_align.align_utils import (load_offsets, offsets_to_coords)
from sbem.render_volume.render_utils import render_volume
from sbem.render_volume.param_schema import RenderVolumeConfig

from datetime import datetime

async def main():
    sbem_experiment="/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731"
    grid_index=1
    start_section=2500
    end_section=9000
    resolution = [11, 11, 33]
    chunk_size=[64, 64, 64]
    downsample = True
    downsample_factors = [8, 8]
    preshift_bits = 6
    minishard_bits = 3


    volume_name = f"s{start_section}_s{end_section}"
    volume_path = os.path.join(sbem_experiment, "volume", volume_name+"_ng")

    offset_dir = os.path.join(sbem_experiment, "zalign", "ob_substack")


    load_sections_spec = dict(sbem_experiment=sbem_experiment,
                              grid_index=grid_index,
                              start_section=, end_section,
                              exclude_sections=exclude_sections)
    load_sections_config = LoadSectionsConfig.from_dict(config["load_sections"])
    xy_offsets, sections = load_offsets(offset_dir, load_sections_config)
    xy_coords = offsets_to_coords(xy_offsets)

    coord_dir = os.path.join(sbem_experiment, "zalign", "ob_substack", "coord")
    if not os.path.exists(coord_dir):
        os.mkdir(coord_dir)
    coord_file = os.path.join(coord_dir, f"{volume_name}.json")
    section_numbers = [s.section_id[0] for s in sections]
    coord_result = dict(section_numbers=section_numbers,
                        xy_offsets=xy_offsets.tolist(),
                        xy_coords=xy_coords.tolist())
    with open(coord_file, "w") as f:
        json.dump(coord_result, f, indent=4)

    print("Start rendering")
    print(datetime.now().strftime('%H: %M: %S %p'))

    render_volme_config = RenderVolumeConfig(
        volume_path=volume_path,
        sections=sections,
        xy_coords=xy_coords,
        resolution=resolution,
        chunk_size=chunk_size,
        preshift_bits=9
        minishard_bits=6
        downsample=True,
        downsample_factors=[4,4],
        downsample_method="mean")

    volume_spec = await prepare_render_volume(render_volme_config)

    volume = await write_volume()

    volume = await render_volume(volume_path, sections, xy_coords, resolution,
                                 downsample=downsample,
                                 downsample_factors=downsample_factors,
                                 preshift_bits=preshift_bits,
                                 minishard_bits=minishard_bits)

    print("End rendering")
    print(datetime.now().strftime('%H: %M: %S %p'))


if __name__ == "__main__":
    asyncio.run(main())
