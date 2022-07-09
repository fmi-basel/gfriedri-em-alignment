import os

@task()
def load_section_pairs():
    sections = load_sections(**load_sections_config.to_dict(), logger=logger)

    stitched_sections = remove_missing_sections(sections)

    section_pairs = get_section_pairs(loaded)


@task()
def align_section_pair(section_pair, align_config, offset_dir):
    pre_path = get_stitched_path(section_pair[0])
    post_path = get_stitched_path(section_pair[1])
    print(pre_path)
    offset_file = f"{section_pair[0].get_name()}_{section_pair[0].get_name()}.json"
    offset_path = os.path.join(offset_dir, offset_file)
    estimate_offset_and_save(pre_path, post_path,
                             align_config, offset_path)
