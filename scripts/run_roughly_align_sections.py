import argparse
import json

import prefect
from prefect import Flow, Parameter, unmapped
from prefect.executors import LocalDaskExecutor

from sbem.section_align.param_schema import (
    LoadSectionsConfig, AlignSectionsConfig)
from sbem.section_align.align_tasks import (
    load_section_pairs, align_section_pair)

from prefect.utilities.debug import raise_on_exception

from prefect import task

with Flow("Roughly-align-sections") as flow:
    load_sections_config = Parameter("load_sections_config", required=True)
    align_config = Parameter("align_config", required=True)
    offset_dir = Parameter("offset_dir", required=True)
    logger = prefect.context.get("logger")
    logger.info("INFO level log message.")

    section_pairs = load_section_pairs(load_sections_config, offset_dir)

    align_obj = align_section_pair.map(section_pairs,
                                       unmapped(align_config),
                                       unmapped(offset_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    load_sections_config = LoadSectionsConfig.from_dict(config["load_sections"])
    align_config = AlignSectionsConfig.from_dict(config["align_sections"])
    offset_dir = config["output"]["offset_dir"]

    num_workers=6
    flow.executor = LocalDaskExecutor(num_workers=num_workers)

    kwargs = dict(load_sections_config=load_sections_config,
                  align_config=align_config,
                  offset_dir=offset_dir)
    state = flow.run(parameters=kwargs)
