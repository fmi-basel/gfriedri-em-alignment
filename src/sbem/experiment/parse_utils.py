import configparser
import json
import os
from os.path import join
from typing import List

import numpy as np
from tqdm import tqdm


def get_acquisition_config(metadata_path: str):
    """
    Give a path to a SBEM metadata file the corresponding config file is
    loaded.
    :param metadata_path:
    :return: config
    """
    with open(metadata_path) as f:
        session_line = f.readline()
        if not session_line.startswith("SESSION"):
            raise ValueError(
                "Metadata file is not in the correct format. It should start with the SESSION entry. File path: {metadata_path}"
            )
        session_line_trimmed = session_line.replace("SESSION: ", "", 1).replace(
            "'", '"'
        )
        config = json.loads(session_line_trimmed)
    return config


def get_tile_spec_from_SBEMtile(sbem_root_dir: str, tile: dict, resolution_xy: float):
    """
    Extract tile information from tile dict and returns as tile_spec dict.

    tile_spec = {
        "tile_id": ,
        "tile_file": ,
        "x": ,
        "y": ,
        "z":
    }

    :param sbem_root_dir: root directory where the data is stored.
    :param tile: dict containign tile information.
    :param resolution_xy: tile resolution.
    :return: dict with the tile spec.
    """
    tileid_split = tile["tileid"].split(".")
    tile_spec = {
        "tile_id": int(tileid_split[1]),
        "grid_num": int(tileid_split[0]),
        "tile_file": join(sbem_root_dir, tile["filename"]),
        "x": int(np.round(float(tile["glob_x"]) / resolution_xy)),
        "y": int(np.round(float(tile["glob_y"]) / resolution_xy)),
        "z": int(tile["slice_counter"]),
    }
    return tile_spec


def read_tile_metadata(
    sbem_root_dir: str, metadata_path: str, tile_grid_num: int, resolution_xy: float
):
    """
    Parse an SBEM metadata file and return all tile-specs as a list.

    :param sbem_root_dir: root directory where the data is stored.
    :param metadata_path: relative path tot he metadata file.
    :param resolution_xy: tile resolution.
    :return: list of tile-specs.
    """
    content = []
    overlap, tile_size = get_spatial_tile_information(
        conf_file=metadata_path.replace("metadata_", "config_"),
        tile_grid_num=tile_grid_num,
    )
    with open(metadata_path) as f:
        for t in filter(lambda l: l.startswith("TILE"), f.readlines()):
            tile = json.loads(t[6:-1].replace("'", '"'))
            tile_spec = get_tile_spec_from_SBEMtile(sbem_root_dir, tile, resolution_xy)
            tile_spec["overlap"] = overlap
            tile_spec["tile_height"] = tile_size[1]
            tile_spec["tile_width"] = tile_size[0]
            if tile_spec["grid_num"] == tile_grid_num:
                content.append(tile_spec)

    return content


def get_spatial_tile_information(conf_file: str, tile_grid_num: int) -> int:
    config = configparser.ConfigParser()
    config.read(conf_file)
    overlaps = json.loads(config["grids"]["overlap"])
    tile_size = json.loads(config["grids"]["tile_size"])
    return overlaps[tile_grid_num], tile_size[tile_grid_num]


def get_tile_metadata(
    sbem_root_dir: str,
    metadata_files: List[str],
    tile_grid_num: int,
    resolution_xy: float,
):
    """
    Load all tile metadata of a block.

    :param sbem_root_dir: root directory where the data is stored.
    :param metadata_files: list of metadata files.
    :param tile_grid_num: tile grid number.
    :param resolution_xy: tile resolution.
    :return: list of all loaded tile-specs.
    """
    tile_specs = []
    for mf in tqdm(metadata_files, desc="Collect Metadata"):
        if os.stat(mf).st_size == 0:
            continue
        config = get_acquisition_config(mf)
        grid_pixel_size = config["pixel_sizes"][tile_grid_num]
        if grid_pixel_size == resolution_xy:
            tile_specs += read_tile_metadata(
                sbem_root_dir, mf, tile_grid_num, resolution_xy
            )
        else:
            print("Acquisition parameters changed. Only returning first stack.")
            return tile_specs

    return tile_specs
