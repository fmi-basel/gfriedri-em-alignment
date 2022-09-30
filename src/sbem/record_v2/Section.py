from __future__ import annotations

import json
import os
import warnings
from os.path import exists, join
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from ruyaml import YAML

from sbem.record_v2.Info import Info

if TYPE_CHECKING:
    from typing import Dict

    from sbem.record_v2.Sample import Sample
    from sbem.record_v2.Tile import Tile


class Section(Info):
    def __init__(
        self,
        name: str,
        root: str,
        stitched: bool,
        skip: bool,
        acquisition: str,
        sample: Sample,
        section_num: int,
        tile_grid_num: int,
        thickness: int,
        tile_height: int,
        tile_width: int,
        tile_overlap: int,
        license: str = "Creative Commons Attribution licence (CC " "BY)",
    ):
        super().__init__(name=name, license=license)
        self._root = root
        self._sample = sample
        self._section_num = section_num
        self._tile_grid_num = tile_grid_num
        self._acquisition = acquisition
        self._thickness = thickness
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._tile_overlap = tile_overlap
        self._stitched = stitched
        self._skip = skip
        self._tile_id_map = None
        self.tiles: Dict[int, Tile] = {}
        self._fully_initialized = self._root is None

        if self._sample is not None:
            self._sample.add_section(self)

    def load_from_yaml(self):
        if self._root is not None:
            yaml = YAML(typ="rt")
            with open(self._root) as f:
                dict = yaml.load(f)

            assert self.get_name() == dict["name"]
            assert self.get_format_version() == dict["format_version"]
            assert self._acquisition == dict["acquisition"]
            assert self._stitched == dict["stitched"]
            assert self._skip == dict["skip"]

            super()._license = dict["license"]
            self._section_num = dict["section_num"]
            self._tile_grid_num = dict["tile_grid_num"]
            self._thickness = dict["thickness"]
            self._tile_height = dict["tile_height"]
            self._tile_width = dict["tile_width"]
            self._tile_overlap = dict["tile_overlap"]

            for t_key, t_dict in dict["tiles"].items():
                tile = Tile(
                    section=self,
                    tile_id=t_dict["tile_id"],
                    path=t_dict["path"],
                    stage_x=t_dict["x"],
                    stage_y=t_dict["y"],
                    resolution_xy=t_dict["resolution_xy"],
                )
                self.tiles[t_key] = tile

            self._fully_initialized = True

    def add_tile(self, tile: Tile):
        self.tiles[tile.get_tile_id()] = tile

    def get_tile(self, tile_id: int):
        if tile_id not in self.tiles.keys():
            self.warn_load_yaml("tile_id")
            return None
        return self.tiles[tile_id]

    def warn_load_yaml(self, var_name):
        warnings.warn(
            f"`{var_name}` is `None`. "
            "Load values from yaml with "
            "`section.load_from_yaml()`."
        )

    def get_section_num(self) -> int:
        if self._section_num is None:
            self.warn_load_yaml("section_num")
        return self._section_num

    def get_tile_grid_num(self) -> int:
        if self._tile_grid_num is None:
            self.warn_load_yaml("tile_grid_num")
        return self._tile_grid_num

    def get_thickness(self) -> int:
        if self._thickness is None:
            self.warn_load_yaml("thickness")
        return self._thickness

    def get_acquisition(self) -> str:
        return self._acquisition

    def is_stitched(self) -> bool:
        return self._stitched

    def skip(self) -> bool:
        return self._skip

    def get_sample(self) -> Sample:
        return self._sample

    def _compute_tile_id_map(self) -> ArrayLike:
        xx, yy = set(), set()
        coords_to_tile = {}
        for t in self.tiles.values():
            x, y = int(t.x), int(t.y)
            if x in coords_to_tile.keys():
                coords_to_tile[x][y] = t.get_tile_id()
            else:
                coords_to_tile[x] = {y: t.get_tile_id()}
            xx.add(x)
            yy.add(y)

        xx = list(sorted(xx))
        yy = list(sorted(yy))

        tile_id_map = [[]]
        for j_map, j in enumerate(
            range(yy[0], yy[-1] + 1, self._tile_height - self._tile_overlap)
        ):
            for i in range(xx[0], xx[-1] + 1, self._tile_width - self._tile_overlap):

                tile_id = -1
                if i in coords_to_tile.keys():
                    if j in coords_to_tile[i].keys():
                        tile_id = coords_to_tile[i][j]

                if len(tile_id_map) <= j_map:
                    tile_id_map.append([tile_id])
                else:
                    tile_id_map[j_map].append(tile_id)

        return np.array(tile_id_map)

    def get_tile_id_map(self, path=None) -> ArrayLike:
        if path is not None and exists(path):
            # Load from disk
            with open(path) as f:
                data = json.load(f)
                return np.array(data)
        elif path is not None and not exists(path):
            # Compute and save to disk
            tile_id_map = self._compute_tile_id_map()
            with open(path, "w") as f:
                json.dump(tile_id_map.tolist(), f)
        else:
            # Just compute
            return self._compute_tile_id_map()

    def to_dict(self) -> Dict:
        tiles = {}
        for k in self.tiles.keys():
            t = self.tiles.get(k)
            tiles[str(t.get_tile_id())] = t.to_dict()

        return {
            "license": self.get_license(),
            "format_version": self.get_format_version(),
            "section_num": self._section_num,
            "tile_grid_num": self._tile_grid_num,
            "thickness": self._thickness,
            "tile_height": self._tile_height,
            "tile_width": self._tile_width,
            "tile_overlap": self._tile_overlap,
            "tile_id_map": self._tile_id_map,
            "tiles": tiles,
        }

    def get_section_id(self) -> str:
        return f"s{self.get_section_num()}_g{self._tile_grid_num}"

    def _dump(self, path: str):
        yaml = YAML(typ="rt")
        with open(join(path, "section.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f)

    def save(self, path: str, overwrite: bool = False):
        out_path = join(path, self.get_section_id())
        if not exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            self._dump(path=out_path)
        else:
            if overwrite:
                self._dump(path=out_path)
            else:
                raise FileExistsError()
