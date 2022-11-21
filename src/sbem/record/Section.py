from __future__ import annotations

import json
import os
import shutil
from os.path import exists, join
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import ArrayLike
from ruyaml import YAML

from sbem.record.Info import Info
from sbem.record.Tile import Tile

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict

    from sbem.record.Sample import Sample


class Section(Info):
    def __init__(
        self,
        sample: Sample,
        name: str,
        stitched: bool,
        skip: bool,
        acquisition: str,
        section_num: int,
        tile_grid_num: int,
        thickness: float,
        tile_height: int,
        tile_width: int,
        tile_overlap: int,
        alignment_mesh: str = None,
        license: str = "Creative Commons Attribution licence (CC " "BY)",
    ):
        super().__init__(name=name, license=license)
        self.set_sample(sample)
        self._section_num = section_num
        self._tile_grid_num = tile_grid_num
        self._acquisition = acquisition
        self._thickness = thickness
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._tile_overlap = tile_overlap
        self._stitched = stitched
        self._skip = skip
        self._alignment_mesh = alignment_mesh
        self.tiles: Dict[int, Tile] = {}
        self._fully_initialized = True

        if self._sample is not None:
            self._sample.add_section(self)

    class _Decorator:
        def is_initialized(func):
            def wrapper(self, *args, **kwargs):
                if not self._fully_initialized:
                    raise RuntimeError(
                        "Section is not fully initialized. Load "
                        "from yaml with `section.load_from_yaml("
                        "path)`."
                    )
                return func(self, *args, **kwargs)

            return wrapper

    @_Decorator.is_initialized
    def add_tile(self, tile: Tile):
        if tile.get_section() is None:
            tile.set_section(self)
        else:
            assert tile.get_section() == self, "Tile belongs to another section."
        self.tiles[tile.get_tile_id()] = tile

    @_Decorator.is_initialized
    def get_tile(self, tile_id: int):
        if tile_id in self.tiles.keys():
            return self.tiles[tile_id]
        else:
            return None

    def get_section_num(self) -> int:
        return self._section_num

    def get_tile_grid_num(self) -> int:
        return self._tile_grid_num

    @_Decorator.is_initialized
    def get_thickness(self) -> float:
        return self._thickness

    @_Decorator.is_initialized
    def get_tile_height(self) -> int:
        return self._tile_height

    @_Decorator.is_initialized
    def get_tile_width(self) -> int:
        return self._tile_width

    @_Decorator.is_initialized
    def get_tile_overlap(self) -> int:
        return self._tile_overlap

    def get_acquisition(self) -> str:
        return self._acquisition

    def is_stitched(self) -> bool:
        return self._stitched

    def skip(self) -> bool:
        return self._skip

    def set_sample(self, sample: Sample):
        self._sample = sample

    def get_sample(self) -> Sample:
        return self._sample

    @_Decorator.is_initialized
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

    @_Decorator.is_initialized
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

    @_Decorator.is_initialized
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

    def to_dict(self) -> Dict:
        if self._fully_initialized:
            tiles = []
            for t in self.tiles.values():
                tiles.append(t.to_dict())

            return {
                "license": self.get_license(),
                "format_version": self.get_format_version(),
                "thickness": self._thickness,
                "tile_height": self._tile_height,
                "tile_width": self._tile_width,
                "tile_overlap": self._tile_overlap,
                "alignment_mesh": self._alignment_mesh,
                "tiles": tiles,
            }
        else:
            return {
                "license": None,
                "format_version": self.get_format_version(),
                "thickness": None,
                "tile_height": None,
                "tile_width": None,
                "tile_overlap": None,
                "alignment_mesh": None,
                "tiles": [],
            }

    def _dump(self, path: str):
        yaml = YAML(typ="rt")
        with open(join(path, "section.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f)

    def save(self, path: str, overwrite: bool = False):
        out_path = join(path, self.get_name())
        if not exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            self._dump(path=out_path)
        else:
            if overwrite:
                if self._fully_initialized:
                    self._dump(path=out_path)
            else:
                raise FileExistsError()

    def get_section_dir(self):
        sample_exists = self.get_sample() is not None
        exp_exists = self.get_sample().get_experiment() is not None
        if sample_exists and exp_exists:
            return join(
                self.get_sample().get_experiment().get_root_dir(),
                self.get_sample().get_experiment().get_name(),
                self.get_sample().get_name(),
                self.get_name(),
            )
        else:
            return None

    def load_from_yaml(self, path: str = None):
        if path is None:
            sample_exists = self.get_sample() is not None
            exp_exists = self.get_sample().get_experiment() is not None
            if sample_exists and exp_exists:
                path = join(
                    self.get_sample().get_experiment().get_root_dir(),
                    self.get_sample().get_experiment().get_name(),
                    self.get_sample().get_name(),
                    self.get_name(),
                    "section.yaml",
                )

        if path is not None:
            yaml = YAML(typ="rt")
            with open(path) as f:
                dict = yaml.load(f)

            self._load_details(dict)

    def _load_details(self, dict: Dict):
        assert self.get_format_version() == dict["format_version"]

        self._license = dict["license"]
        self._thickness = dict["thickness"]
        self._tile_height = dict["tile_height"]
        self._tile_width = dict["tile_width"]
        self._tile_overlap = dict["tile_overlap"]
        self._alignment_mesh = dict["alignment_mesh"]

        self._fully_initialized = True

        for t_dict in dict["tiles"]:
            tile = Tile(
                section=self,
                tile_id=t_dict["tile_id"],
                path=t_dict["path"],
                stage_x=t_dict["stage_x"],
                stage_y=t_dict["stage_y"],
                resolution_xy=t_dict["resolution_xy"],
            )
            self.tiles[tile.get_tile_id()] = tile

    @staticmethod
    def lazy_loading(
        name: str,
        section_num: int,
        tile_grid_num: int,
        stitched: bool,
        skip: bool,
        acquisition: str,
        details: Union[Dict, str],
    ) -> Section:
        sec = Section(
            name=name,
            stitched=stitched,
            skip=skip,
            acquisition=acquisition,
            sample=None,
            section_num=section_num,
            tile_grid_num=tile_grid_num,
            thickness=None,
            tile_height=None,
            tile_width=None,
            tile_overlap=None,
            alignment_mesh=None,
            license=None,
        )

        if isinstance(details, str):
            sec._fully_initialized = False
        else:
            sec._load_details(details)

        return sec

    def get_alignment_mesh(self) -> str:
        return self._alignment_mesh

    def set_alignment_mesh(self, path: str):
        self._alignment_mesh = path

    def delete_dir(self):
        section_dir = self.get_section_dir()
        shutil.rmtree(section_dir)
