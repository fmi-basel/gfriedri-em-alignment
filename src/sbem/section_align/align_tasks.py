import os
import traceback

import caching
from caching import task

from sbem.section_align.align_utils import (
    estimate_offset_and_save,
    get_pair_name,
    get_section_pairs,
    load_sections,
    log_missing_sections,
    remove_missing_sections,
)
from sbem.section_align.param_schema import LoadSectionsConfig


@task()
def load_section_pairs(load_sections_config: LoadSectionsConfig, offset_dir: str):
    logger = caching.context.get("logger")
    sections = load_sections(**load_sections_config.to_dict(), logger=logger)
    stitched, unloaded, unstitched = remove_missing_sections(sections)
    log_missing_sections(stitched, unloaded, unstitched, offset_dir)
    section_pairs = get_section_pairs(stitched)
    return section_pairs


@task()
def align_section_pair(section_pair, align_config, offset_dir, debug=False):
    logger = caching.context.get("logger")
    pair_name = get_pair_name(section_pair)
    offset_file = f"{pair_name}.json"
    offset_path = os.path.join(offset_dir, offset_file)
    logger.info(f"Aligning sections {pair_name}")

    try:
        pre_path = section_pair[0].get_stitched_path()
        post_path = section_pair[1].get_stitched_path()
        estimate_offset_and_save(
            pre_path, post_path, align_config, offset_path, debug=debug
        )
    except Exception as e:
        with open(offset_file, "w") as f:
            f.write('"error"')
        logger.error(f"Encounter error in section pair {pair_name}.")
        logger.error(e)
        tb = traceback.format_exc()
        logger.error(tb)
