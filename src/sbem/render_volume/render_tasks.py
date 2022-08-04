import logging

from sbem.section_align import load_offsets, offsets_to_coords, save_coords
from sbem.render_utils import (
    load_stitched_and_prepare_volume,
    prepare_sharding,
    create_volume,
    write_volume)


async def render_volume(load_sections_config, coord_file, volume_config,
                        downsample_config=None,
                        write=True,
                        logger=logging.getLogger()):
    xy_offsets, sections = load_offsets(offset_dir, load_sections_config)
    xy_coords = offsets_to_coords(xy_offsets)

    save_coords(coord_file, sections, xy_offsets, xy_coords)


    stitched_sections, xy_coord, size_hierarchy = \
    await load_stitched_and_prepare_volume(sections, xy_coords,
                                           volume_config.chunk_size,
                                           downsample_config)

    sharding_spec, size_hierarchy = prepare_sharding(size_hiearchy,
                                                     volume_config.preshift_bits,
                                                     volume_config.minishard_bits)

    if write:
        logger.info("Start rendering")
        start_time = datetime.now()
        logger.info(start_time.strftime('%H: %M: %S %p'))

        volume = await create_volume(volume_path,
                                     size_hierarchy.volume_size,
                                     volume_config.chunk_size,
                                     volume_config.resolution,
                                     sharding=True,
                                     sharding_spec=sharding_spec)
        await write_volume(volume, stitched_sections, xy_coords, size_hiearchy)
        logger.info("End rendering")
        end_time = datetime.now()
        logger.info(end_time.strftime('%H: %M: %S %p'))
        excution_time = end_time - start_time
        logger.info(f"Excution time: {execution_time.strftime('%H: %M: %S %p')}")
    else:
        logger.info("Prepare volume:")
        logger.info(f"size_hiearchy: {size_hiearchy.to_dict()}"
        logger.info(f"sharding_spec: {sharding_spec}")
