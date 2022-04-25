import json
from configparser import ConfigParser
from os.path import join, split
from typing import List

from tqdm import tqdm


def get_acquisition_config(metadata_path: str):
    dir_, name_ = split(metadata_path)
    conf_path = join(dir_, name_.replace("metadata", "config"))
    config = ConfigParser()
    config.read(conf_path)
    return config


def get_tile_spec_from_SBEMtile(exp_path: str, tile: dict, resolution_xy: float):
    tile_spec = {
        "tile_id": int(tile["tileid"].split(".")[1]),
        "tile_file": join(exp_path, tile["filename"]),
        "x": float(tile["glob_x"]) // resolution_xy,
        "y": float(tile["glob_y"]) // resolution_xy,
        "z": int(tile["slice_counter"]),
    }
    return tile_spec


def read_tile_metadata(exp_path: str, metadata_path: str, resolution_xy: float):
    content = []
    with open(metadata_path) as f:
        for t in filter(lambda l: l.startswith("TILE"), f.readlines()):
            tile = json.loads(t[6:-1].replace("'", '"'))
            content.append(get_tile_spec_from_SBEMtile(exp_path, tile, resolution_xy))

    return content


def get_tile_metadata(
    exp_path: str, metadata_files: List[str], tile_grid_num: int, resolution_xy: float
):
    tile_specs = []
    for mf in tqdm(metadata_files, desc="Collect Metadata"):
        config = get_acquisition_config(mf)
        grid_is_active = json.loads(config["grids"]["grid_active"])[tile_grid_num]
        grid_pixel_size = json.loads(config["grids"]["pixel_size"])[tile_grid_num]
        if grid_is_active and grid_pixel_size == resolution_xy:
            tile_specs += read_tile_metadata(exp_path, mf, resolution_xy)
        else:
            print("Acquisition parameters changed. Only returning first stack.")
            return tile_specs

    return tile_specs
