import os
import sys
import asyncio
import json
import argparse
import logging

from sbem.render_volume.render_tasks import render_volume
from sbem.render_volume.schema import VolumeConfig, DownsampleConfig
from sbem.section_align.param_schema import LoadSectionsConfig


from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--no_write", action="store_true",
                        help="Do not write the volume data to disk")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing volume")
    args = parser.parse_args()

    logger = logging.getLogger("run_render_volume")
    logger.setLevel(logging.DEBUG)
    logger_sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(logger_sh)

    with open(args.config) as f:
        config = json.load(f)

    load_sections_config = LoadSectionsConfig.from_dict(config["load_sections"])

    offset_dir = config["offset_dir"]
    coord_file = config["coord_file"]

    volume_config = VolumeConfig.from_dict(config["volume"])

    if "downsample" in config:
        downsample_config = DownsampleConfig.from_dict(config["downsample"])
    else:
        downsample_config = None

    if "write_range" in config:
        write_range = config["write_range"]
    else:
        write_range = None

    if "write_to_existing" in config:
        write_to_existing = config["write_to_existing"]
    else:
        write_to_existing = False

    kwargs = dict(load_sections_config=load_sections_config,
                  offset_dir=offset_dir,
                  coord_file=coord_file,
                  volume_config=volume_config,
                  downsample_config=downsample_config,
                  overwrite=args.overwrite,
                  no_write=args.no_write,
                  write_range=write_range,
                  write_to_existing=write_to_existing,
                  logger=logger)

    asyncio.run(render_volume(**kwargs))
