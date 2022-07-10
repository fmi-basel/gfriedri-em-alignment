import os
import prefect
from prefect import task

from sbem.section_align.align_utils import (
    load_sections, remove_missing_sections,
    log_missing_sections, get_section_pairs,
    estimate_offset_and_save)
from sbem.section_align.param_schema import (
    LoadSectionsConfig, AlignSectionsConfig)

@task()
def load_section_pairs(load_sections_config: LoadSectionsConfig,
                       offset_dir: str):
    logger = prefect.context.get("logger")
    sections = load_sections(**load_sections_config.to_dict(), logger=logger)
    stitched, unloaded, unstitched = remove_missing_sections(sections)
    log_missing_sections(stitched, unloaded, unstitched, offset_dir)
    section_pairs = get_section_pairs(stitched)
    return section_pairs


@task()
def align_section_pair(section_pair, align_config, offset_dir):
    logger = prefect.context.get("logger")
    pair_name = f"{section_pair[0].get_name()}_{section_pair[0].get_name()}"
    logger.info(f"Aligning sections {pair_name}")

    pre_path = section_pair[0].get_stitched_path()
    post_path = section_pair[1].get_stitched_path()
    offset_file = f"{pair_name}.json"
    offset_path = os.path.join(offset_dir, offset_file)
    estimate_offset_and_save(pre_path, post_path,
                             align_config, offset_path)
