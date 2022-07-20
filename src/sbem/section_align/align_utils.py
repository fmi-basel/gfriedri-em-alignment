import os
import json
import zarr
import numpy as np

from skimage.transform import downscale_local_mean
from sofima import stitch_rigid
from sbem.experiment import Experiment


def load_json(path):
    with open(path) as f:
        x = json.load(f)
    return x


def load_n5(path):
    img = zarr.open(zarr.N5FSStore(path), mode="r")
    return img


def downscale_image(img, factor):
    return downscale_local_mean(img, factor)


def crop_image_center(img, dx, dy):
    img_shape = img.shape
    center = np.array(img_shape) // 2
    cropped = img[center[0]-dx:center[0]+dx,
            center[1]-dy:center[1]+dy]
    return cropped


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

    pre_cropped = crop_image_center(pre, *align_config.crop_size)
    post_cropped = crop_image_center(post, *align_config.crop_size)

    # dsf = tuple([align_config.downscale_factor] * 2)
    dsf = align_config.downscale_factor
    pre_scaled = downscale_image(pre_cropped, dsf)
    post_scaled = downscale_image(post_cropped, dsf)

    xyo, pr = estimate_offset(pre_scaled, post_scaled, align_config)
    save_offset(xyo, pr, offset_path)


def load_sections(sbem_experiment, grid_index, start_section, end_section,
                  logger=None):
    exp = Experiment(logger=logger)
    exp.load(sbem_experiment)
    sections = exp.load_sections(start_section, end_section, grid_index)
    return sections


def log_section_numbers(sections, path):
    section_numbers = [s.section_id[0] for s in sections]
    with open(path, "w") as f:
        json.dump(section_numbers, f)


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
    log_section_numbers(stitched, os.path.join(log_dir, "stitched_sections.json"))
    log_section_numbers(unloaded, os.path.join(log_dir, "unloaded_sections.json"))
    log_section_numbers(unstitched, os.path.join(log_dir, "unstitched_sections.json"))


def get_section_pairs(sections):
    section_pairs = zip(sections[:-1], sections[1:])
    return list(section_pairs)


def get_pair_name(section_pair):
    return f"{section_pair[0].get_name()}_{section_pair[1].get_name()}"


def load_offsets(offset_dir, sbem_experiment, grid_index, start_section, end_section, logger=None):
    sections = load_sections(sbem_experiment, grid_index, start_section, end_section, logger=logger)
    section_pairs = get_section_pairs(sections)

    xyo_list = []
    for pair in section_pairs:
        pair_name = get_pair_name(pair)
        offset_path = os.path.join(offset_dir, f"{pair_name}.json")
        try:
            offset = load_json(offset_path)
        except FileNotFoundError as e:
            print("No offset file found")
            raise e

        if offset == "error":
            msg = f"Alignment for {pair_name} had an error"
            raise ValueError(msg)

        xyo = offset["xyo"]
        xyo_list.append(xyo)

    xy_offsets = np.array(xyo_list)
    return xy_offsets, sections


def offsets_to_coords(xy_offsets):
    xy_coords = np.cumsum(xy_offsets, axis=0)
    return xy_coords
