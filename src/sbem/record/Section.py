from __future__ import annotations

import json
import os
from os.path import exists, join
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import ArrayLike
from ruyaml import YAML

from sbem.record.Tile import Tile

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict


class Section:
    def __init__(
        self,
        acquisition: str,
        section_num: int,
        tile_grid_num: int,
        thickness: float,
        tile_height: int,
        tile_width: int,
        tile_overlap: int,
    ):
        self._section_num = section_num
        self._tile_grid_num = tile_grid_num
        self._acquisition = acquisition
        self._thickness = thickness
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._tile_overlap = tile_overlap
        self.tiles: Dict[int, Tile] = {}

    def add_tile(self, tile: Tile):
        if tile.get_section() is None:
            tile.set_section(self)
        else:
            assert tile.get_section() == self, "Tile belongs to another section."
        self.tiles[tile.get_tile_id()] = tile

    def get_tile(self, tile_id: int):
        if tile_id in self.tiles.keys():
            return self.tiles[tile_id]
        else:
            return None

    def get_section_num(self) -> int:
        return self._section_num

    def get_tile_grid_num(self) -> int:
        return self._tile_grid_num

    def get_thickness(self) -> float:
        return self._thickness

    def get_tile_height(self) -> int:
        return self._tile_height

    def get_tile_width(self) -> int:
        return self._tile_width

    def get_tile_overlap(self) -> int:
        return self._tile_overlap

    def get_acquisition(self) -> str:
        return self._acquisition

    def _compute_tile_id_map(self) -> ArrayLike:
        if len(self.tiles) == 0:
            return None

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

    def get_tile_id_map(self, path: str = None) -> ArrayLike:
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

            return tile_id_map
        else:
            # Just compute
            return self._compute_tile_id_map()

    def get_tile_data_map(self, path: str = None, indexing="yx"):
        """
        Get a tile-data-map mapping tile (x, y) coordinates to the loaded
        image data.

        :return: tile-data-map
        """
        assert indexing == "xy" or indexing == "yx"
        tile_id_map = self.get_tile_id_map(path=path)
        tile_data_map = {}
        for y in range(tile_id_map.shape[0]):
            for x in range(tile_id_map.shape[1]):
                if tile_id_map[y, x] != -1:
                    data = self.tiles[tile_id_map[y, x]].get_tile_data()
                    if indexing == "xy":
                        tile_data_map[(x, y)] = data
                    else:
                        tile_data_map[(y, x)] = data
        return tile_data_map

    def set_coarse_offsets(self, cx: ArrayLike, cy: ArrayLike, path: str) -> None:
        """
        Save coarse offsets (cx, cy) to json.

        :param cx:
            Coarse offset cx.
        :param cy:
            Coarse offset cy.
        :param path:
            Where to save the offsets.
        """
        cx_cy = dict(cx=cx.tolist(), cy=cy.tolist())
        with open(path, "w") as f:
            json.dump(cx_cy, f)

    def get_coarse_offsets(self, path=None) -> Union[None, tuple[ArrayLike, ArrayLike]]:
        """
        Load coarse offsets.

        If no file exists None is returned.

        :param path:
            Where the coarse offset is stored.
        :return:
            cx, cy corase offsets.
        """
        if path is None:
            return None
        else:
            if exists(path):
                with open(path) as f:
                    cx_cy = json.load(f)

                cx = np.array(cx_cy["cx"])
                cy = np.array(cx_cy["cy"])

                return cx, cy
            else:
                return None

    def to_dict(self) -> Dict:
        tiles = []
        for t in self.tiles.values():
            tiles.append(t.to_dict())

        return {
            "section_num": self._section_num,
            "tile_grid_num": self._tile_grid_num,
            "acquisition": self._acquisition,
            "thickness": self._thickness,
            "tile_height": self._tile_height,
            "tile_width": self._tile_width,
            "tile_overlap": self._tile_overlap,
            "tiles": tiles,
        }

    def _dump(self, path: str):
        yaml = YAML(typ="rt")
        with open(join(path, "section.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f)

    def get_name(self):
        return f"s{self._section_num}_g{self._tile_grid_num}"

    def save(self, path: str, overwrite: bool = False):
        out_path = join(path, self.get_name())
        if not exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            self._dump(path=out_path)
        else:
            if overwrite:
                self._dump(path=out_path)
            else:
                raise FileExistsError()

    @classmethod
    def load_from_yaml(cls, path: str):
        yaml = YAML(typ="rt")
        with open(path) as f:
            dict = yaml.load(f)

        section = cls(
            section_num=dict["section_num"],
            tile_grid_num=dict["tile_grid_num"],
            acquisition=dict["acquisition"],
            thickness=dict["thickness"],
            tile_height=dict["tile_height"],
            tile_width=dict["tile_width"],
            tile_overlap=dict["tile_overlap"],
        )

        for t_dict in dict["tiles"]:
            tile = Tile(
                section=section,
                tile_id=t_dict["tile_id"],
                path=t_dict["path"],
                stage_x=t_dict["stage_x"],
                stage_y=t_dict["stage_y"],
                resolution_xy=t_dict["resolution_xy"],
            )
            section.tiles[tile.get_tile_id()] = tile

        return section
