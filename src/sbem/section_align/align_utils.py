import os
import json
import zarr


from skimage.transform import downscale_local_mean
from sofima import stitch_rigid
from sbem.experiment import Experiment


def load_n5(path):
    img = zarr.open(zarr.N5FSStore(path), mode="r")
    return img


def downscale_image(img, factor):
    return downscale_local_mean(img, factor)


def estimate_offset(pre, post, align_config):
    xyo, pr = stitch_rigid._estimate_offset(pre, post,
                                            align_config.range_limit,
                                            align_config.filter_size)
    return xyo, pr


def save_offset(xyo, pr, save_path):
    result = dict(xyo=list(map(float, xyo)),
                  pr=float(pr))
    with open(save_path, "w") as f:
        json.dump(result, f)


def estimate_offset_and_save(pre_path, post_path, align_config, offset_path):
    pre = load_n5(pre_path)
    post = load_n5(post_path)
    dsf = tuple([align_config.downscale_factor] * 2)
    pre_scaled = downscale_image(pre, dsf)
    post_scaled = downscale_image(post, dsf)
    xyo, pr = estimate_offset(pre_scaled, post_scaled, align_config)
    save_offset(xyo, pr, offset_path)


def load_sections(sbem_experiment, grid_index, start_section, end_section,
                  logger=None):
    exp = Experiment(logger=logger)
    exp.load(sbem_experiment)
    sections = exp.load_sections(start_section, end_section, grid_index)
    return sections


def log_section_ids(sections, path):
    section_ids = [s.section_id for s in sections]
    with open(path, "w") as f:
        json.dump(section_ids, f)


def remove_missing_sections(sections):
    stitched = []
    unloaded  = []
    unstitched = []
    for sec in sections:
        if sec is None:
            continue # TODO log non-existing sections too
        elif not sec.is_loaded:
            unloaded.append(sec)
        elif not sec.check_stitched():
            unstitched.append(sec)
        else:
            stitched.append(sec)


    return stitched, unloaded, unstitched


def log_missing_sections(stitched, unloaded, unstitched, log_dir):
    log_section_ids(stitched, os.path.join(log_dir, "stitched_sections.json"))
    log_section_ids(unloaded, os.path.join(log_dir, "unloaded_sections.json"))
    log_section_ids(unstitched, os.path.join(log_dir, "unstitched_sections.json"))


def get_section_pairs(sections):
    section_pairs = zip(sections[1:], sections[:-2])
    return list(section_pairs)
