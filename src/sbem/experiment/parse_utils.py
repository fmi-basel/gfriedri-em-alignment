import json
from configparser import ConfigParser
from os.path import join, split
from typing import List

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
            raise ValueError("Metadata file is not in the correct format. It should start with the SESSION entry. File path: {metadata_path}")
        session_line_trimmed = session_line.replace("SESSION: ", "", 1).replace("'", '"')
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
    tile_spec = {
        "tile_id": int(tile["tileid"].split(".")[1]),
        "tile_file": join(sbem_root_dir, tile["filename"]),
        "x": float(tile["glob_x"]) // resolution_xy,
        "y": float(tile["glob_y"]) // resolution_xy,
        "z": int(tile["slice_counter"]),
    }
    return tile_spec


def read_tile_metadata(sbem_root_dir: str, metadata_path: str, resolution_xy: float):
    """
    Parse an SBEM metadata file and return all tile-specs as a list.

    :param sbem_root_dir: root directory where the data is stored.
    :param metadata_path: relative path tot he metadata file.
    :param resolution_xy: tile resolution.
    :return: list of tile-specs.
    """
    content = []
    with open(metadata_path) as f:
        for t in filter(lambda l: l.startswith("TILE"), f.readlines()):
            tile = json.loads(t[6:-1].replace("'", '"'))
            content.append(
                get_tile_spec_from_SBEMtile(sbem_root_dir, tile, resolution_xy)
            )

    return content


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
        config = get_acquisition_config(mf)
        grid_pixel_size = config["pixel_sizes"][tile_grid_num]
        if grid_pixel_size == resolution_xy:
            tile_specs += read_tile_metadata(sbem_root_dir, mf, resolution_xy)
        else:
            print("Acquisition parameters changed. Only returning first stack.")
            return tile_specs

    return tile_specs
