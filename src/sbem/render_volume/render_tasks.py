import os
import logging

from datetime import datetime

from sbem.render_volume.render_utils import (
    prepare_volume,
    prepare_existing_volume,
    load_sections_with_range,
    read_stitched_sections,
    write_volume)


async def render_volume(load_sections_config,
                        offset_dir,
                        coord_file,
                        volume_config,
                        downsample_config=None,
                        overwrite=False,
                        no_write=False,
                        write_range=None,
                        write_to_existing=None,
                        logger=logging.getLogger("render_volume")):
    if not write_to_existing:
        volume, stitched_sections, xy_coords, size_hierarchy = \
          await prepare_volume(load_sections_config,
                               offset_dir,
                               coord_file,
                               volume_config,
                               downsample_config=downsample_config,
                               overwrite=overwrite,
                               logger=logger)
        if write_range is None:
            write_sections = stitched_sections
            write_xy_coords = xy_coords
            start_z = 0
        else:
            write_sections = stitched_sections[write_range[0]:write_range[1]]
            write_xy_coords = xy_coords[write_range[0]:write_range[1]]
            start_z = write_range[0]
    elif write_range is not None:
        volume, size_hierarchy = await prepare_existing_volume(volume_config.path)
        sections, write_xy_coords = \
          load_sections_with_range(load_sections_config, write_range, coord_file)
        write_sections = await read_stitched_sections(sections)
        start_z = write_range[0]
    else:
        raise ValueError("For writing to existing volume, please supply section range.")

    if no_write:
        return volume


    # Write data to volume
    logger.info("Start rendering")
    start_time = datetime.now()
    logger.info(start_time.strftime('%H: %M: %S %p'))
    await write_volume(volume, write_sections, write_xy_coords,
                       size_hierarchy,
                       start_z=start_z)
    logger.info("End rendering")
    end_time = datetime.now()
    logger.info(end_time.strftime('%H: %M: %S %p'))
    execution_time = end_time - start_time
    logger.info(f"Excution time: {execution_time}")
