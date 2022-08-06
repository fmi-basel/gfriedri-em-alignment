import os
import json
import logging
import zarr
import numpy as np

from skimage.transform import downscale_local_mean
from skimage.io import imsave
from sofima import stitch_rigid
from sbem.experiment import Experiment


def load_json(path):
    with open(path) as f:
        x = json.load(f)
    return x


def load_n5(path):
    img = zarr.open(zarr.N5FSStore(path), mode="r")
    return img


def downsample_image(img, factor):
    return downscale_local_mean(img, factor)


def crop_image_center(img, dx, dy):
    img_shape = img.shape
    center = np.array(img_shape) // 2
    cropped = img[center[0]-dy:center[0]+dy,
            center[1]-dx:center[1]+dx]
    return cropped, center


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


def estimate_offset_and_save(pre_path, post_path, align_config, offset_path,
                             debug=False):
    pre = load_n5(pre_path)
    post = load_n5(post_path)

    margin = 10
    done = False
    for crop_size in align_config.crop_sizes:
        pre_cropped, pre_ctr = crop_image_center(pre, *crop_size)
        post_cropped, post_ctr = crop_image_center(post, *crop_size)

        if align_config.downsample:
            dsf = align_config.downsample_factors
            pre_cropped = downsample_image(pre_cropped, dsf)
            post_cropped = downsample_image(post_cropped, dsf)

        if debug:
            base_dir, file_name = os.path.split(offset_path)
            debug_dir = os.path.join(base_dir, f"debug_{os.path.splitext(file_name)[0]}")
            if not os.path.exists(debug_dir):
                os.mkdir(debug_dir)
            imsave(os.path.join(debug_dir, "pre.tif"), pre_cropped)
            imsave(os.path.join(debug_dir, "post.tif"), post_cropped)

        xyo, pr = estimate_offset(pre_cropped, post_cropped, align_config)

        if not np.isnan(xyo).any():
            dist_to_crop = np.array(crop_size) - margin - np.abs(xyo)
            if (dist_to_crop >= 0).all():
                done = True
                break

    if not done:
        raise Exception("No match found.")
    # Offset w.r.t top-let corner of each image
    ctr_diff = pre_ctr - post_ctr
    xyo = xyo + np.flip(ctr_diff)

    if align_config.downsample:
        xyo = np.multiply(xyo, align_config.downsample_factors)
    save_offset(xyo, pr, offset_path)


def load_sections(sbem_experiment, grid_index, start_section, end_section,
                  exclude_sections=None,
                  logger=None):
    if logger is None:
        logger = logging.getLogger()
    exp = Experiment(logger=logger)
    exp.load(sbem_experiment)
    sections = exp.load_sections(start_section, end_section, grid_index)
    if exclude_sections:
        sections = [s for s in sections
                    if s.section_id[0] not in exclude_sections]
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


def load_offsets(offset_dir, load_sections_config):
    sections = load_sections(**load_sections_config.to_dict())
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


def offsets_to_coords(xy_offsets, non_neg=True):
    xy_coords = np.cumsum(xy_offsets, axis=0)
    xy_coords = np.insert(xy_coords, 0, [0,0], axis=0)
    if non_neg:
        xy_coords = xy_coords - xy_coords.min(axis=0)
    xy_coords = xy_coords.astype(int)
    return xy_coords


def save_coords(coord_file, sections, xy_offsets, xy_coords):
    section_numbers = [s.section_id[0] for s in sections]
    coord_result = dict(section_numbers=section_numbers,
                        xy_offsets=xy_offsets.tolist(),
                        xy_coords=xy_coords.tolist())
    with open(coord_file, "w") as f:
        json.dump(coord_result, f, indent=4)
