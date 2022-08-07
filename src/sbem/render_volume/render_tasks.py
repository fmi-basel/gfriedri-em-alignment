import os
import logging

from datetime import datetime

from sbem.section_align.align_utils import load_offsets, offsets_to_coords, save_coords
from sbem.render_volume.render_utils import (
    load_stitched_and_prepare_volume,
    prepare_sharding,
    create_volume,
    write_volume)


async def render_volume(load_sections_config,
                        offset_dir,
                        coord_file,
                        volume_config,
                        downsample_config=None,
                        overwrite=False,
                        no_write=False,
                        logger=logging.getLogger("render_volume")):
    xy_offsets, sections = load_offsets(offset_dir, load_sections_config)
    xy_coords = offsets_to_coords(xy_offsets)

    save_coords(coord_file, sections, xy_offsets, xy_coords)


    stitched_sections, xy_coords, size_hierarchy = \
    await load_stitched_and_prepare_volume(sections, xy_coords,
                                           volume_config.chunk_size,
                                           downsample_config)

    sharding_spec, size_hierarchy = prepare_sharding(size_hierarchy,
                                                     volume_config.preshift_bits,
                                                     volume_config.minishard_bits,)

    logger.info("Prepare volume:")
    logger.info(f"size_hiearchy: {size_hierarchy.to_dict()}")
    logger.info(f"sharding_spec: {sharding_spec}")

    if os.path.exists(volume_config.path) and not overwrite:
        raise OSError(f"Volume {volume_config.path} already exists. "+\
                      "Please use --overwrite option to overwrite.")

    if not no_write:
        # Create and write data to volume
        logger.info("Start rendering")
        start_time = datetime.now()
        logger.info(start_time.strftime('%H: %M: %S %p'))

        volume = await create_volume(volume_config.path,
                                     size_hierarchy.volume_size,
                                     volume_config.chunk_size,
                                     volume_config.resolution,
                                     sharding=True,
                                     sharding_spec=sharding_spec)
        await write_volume(volume, stitched_sections, xy_coords, size_hierarchy)
        logger.info("End rendering")
        end_time = datetime.now()
        logger.info(end_time.strftime('%H: %M: %S %p'))
        execution_time = end_time - start_time
        logger.info(f"Excution time: {execution_time}")
