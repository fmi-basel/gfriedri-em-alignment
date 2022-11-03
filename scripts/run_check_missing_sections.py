import argparse
import configparser

import caching
from caching import Flow, Parameter
from caching.executors import LocalDaskExecutor

from sbem.section_align.align_tasks import load_section_pairs
from sbem.section_align.param_schema import LoadSectionsConfig

with Flow("Check-missing-sections") as flow:
    load_sections_config = Parameter("load_sections_config", required=True)
    offset_dir = Parameter("offset_dir", required=True)
    logger = caching.context.get("logger")
    logger.info("INFO level log message.")

    section_pairs = load_section_pairs(load_sections_config, offset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    load_sections_config = LoadSectionsConfig.from_dict(dict(config["LOAD_SECTIONS"]))
    offset_dir = config["OUTPUT"]["offset_dir"]

    flow.executor = LocalDaskExecutor()

    kwargs = dict(load_sections_config=load_sections_config, offset_dir=offset_dir)
    state = flow.run(parameters=kwargs)
