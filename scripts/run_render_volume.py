import os
import asyncio
import logging

from sbem.section_align.align_utils import (load_offsets, offsets_to_coords)
from sbem.render_volume.render_utils import render_volume

async def main():
    sbem_experiment="/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731"
    grid_index=1
    start_section=5000
    end_section=5004
    resolution = [11, 11, 33]
    volume_name = f"s{start_section}_s{end_section}"
    volume_path = os.path.join(sbem_experiment, "volume", volume_name+"_ng")

    offset_dir = os.path.join(sbem_experiment, "zalign", "ob_substack")
    logger = logging.getLogger(__name__)
    xy_offsets, sections = load_offsets(offset_dir, sbem_experiment, grid_index,
                                        start_section, end_section, logger=logger)
    xy_coords = offsets_to_coords(xy_offsets)

    volume = await render_volume(volume_path, sections, xy_coords, resolution)


if __name__ == "__main__":
    asyncio.run(main())
