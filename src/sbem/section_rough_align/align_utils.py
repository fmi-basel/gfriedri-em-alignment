import os
import json
import logging
import zarr
from ome_zarr.io import parse_url
import numpy as np

from skimage.transform import downscale_local_mean
from skimage.io import imsave
from sofima import stitch_rigid
from sbem.experiment import Experiment
from sbem.tile_stitching.sofima_utils import load_sections


def load_json(path):
    with open(path) as f:
        x = json.load(f)
    return x


def load_n5(path):
    img = zarr.open(zarr.N5FSStore(path), mode="r")
    return img


def load_from_store(store_dict):
    store = parse_url(store_dict["data_path"], mode="r").store
    zarr_root = zarr.group(store=store)
    z, y, x = store_dict['offset'] + store_dict['origin']
    zs, ys, xs = store_dict['shape']
    img = zarr_root["0"][z : z + zs, y : y + ys, x : x + xs]
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


def estimate_offset_and_save(pre_store, post_store, align_config, offset_path,
                             margin=50,
                             min_diff_thresh=40,
                             debug=False):

    def _is_valid_offset(offset, crop_size):
        dist_to_crop = np.array(crop_size)*2 - margin - np.abs(xyo)
        return (not np.isnan(xyo).any()) and (dist_to_crop >= 0).all()

    # pre = load_n5(pre_path)
    # post = load_n5(post_path)

    pre = load_from_store(pre_store)
    post = load_from_store(post_store)

    pre = np.squeeze(pre)
    post = np.squeeze(post)

    done = False
    max_idx = -1
    max_pr = 0
    estimates = []
    prs = []
    crop_size_list = []
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

        if debug:
            print("xyo, pr, crop_size")
            print(xyo, pr, crop_size)

        estimates.append(xyo)
        prs.append(pr)
        crop_size_list.append(crop_size)

        valid = _is_valid_offset(xyo, crop_size)

        # If a single peak is found, terminate search.
        if pr == 0.0 and valid:
            done = True
            break

        if pr > max_pr and valid:
          max_pr = pr
          max_idx = len(estimates) - 1

    if not done:
        min_diff = np.inf
        min_idx = 0
        for i, (off0, off1) in enumerate(zip(estimates, estimates[1:])):
            diff = np.linalg.norm(np.array(off1) - np.array(off0))
            if diff < min_diff and _is_valid_offset(off1, crop_size_list[i+1]):
                min_diff = diff
                min_idx = i

        # If we found an offset with good consistency between two consecutive
        # estimates, perfer that.
        if min_diff < min_diff_thresh:
            xyo = estimates[min_idx + 1]
            pr = prs[min_idx + 1]
            done = True
        # Otherwise prefer the offset with maximum peak ratio.
        elif max_idx >= 0:
            xyo = estimates[max_idx]
            pr = prs[max_idx]
            done = True

    if not done:
        raise Exception("No match found.")
    # Offset w.r.t top-let corner of each image
    ctr_diff = pre_ctr - post_ctr
    xyo = xyo + np.flip(ctr_diff)

    if debug:
        print(f"center diff: {ctr_diff}")
        print(f"xyo: {xyo}")

    if align_config.downsample:
        xyo = np.multiply(xyo, align_config.downsample_factors)
    save_offset(xyo, pr, offset_path)


def log_section_numbers(sections, path):
    section_numbers = [s.section_id[0] for s in sections]
    with open(path, "w") as f:
        json.dump(section_numbers, f)


def get_section_pairs(sections):
    section_pairs = zip(sections[:-1], sections[1:])
    return list(section_pairs)


def get_pair_name(section_pair):
    return f"{section_pair[0]['section_num']}_{section_pair[1]['section_num']}"
    # return f"{section_pair[0].get_name()}_{section_pair[1].get_name()}"


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


def load_coords(coord_file):
    with open(coord_file) as f:
        coord_result = json.load(f)
    return coord_result



def find_section_store(section_dict, sec_store, logger=None):
    section_num = section_dict["section_num"]
    if section_num not in sec_store.keys():
        msg = (f"Section {section_dict['section_name']} not found "
               f"in volumes {';'.join(volume_paths)}")
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    else:
        sec_store[section_num].update(section_num=section_num)
        return sec_store[section_num]


def summarize_sections_in_volumes(volumes, logger=None):
    offset_maps = [v.get_section_offset_map() for v in volumes]
    origins = [v.get_origin() for v in volumes]
    volume_dirs = [v.get_dir() for v in volumes]

    sec_store = dict()
    for v in volumes:
        offset_map = v.get_section_offset_map()
        shape_map = v.get_section_shape_map()
        origin = v.get_origin()
        data_path = v.get_data_path()

        for key, val in offset_map.items():
            if key not in sec_store:
                sec_store[key] = dict(data_path=data_path,
                                      origin=origin,
                                      offset=val,
                                      shape=shape_map[key])
            else:
                msg = (f"Section {key} appeared at least in multiple volumes."
                       "The first volume will be returned")
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)

    return sec_store
