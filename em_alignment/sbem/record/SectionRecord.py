import json
from logging import Logger
from os import mkdir
from os.path import exists, join

import numpy as np
from sbem.record import BlockRecord
from sbem.record.TileRecord import TileRecord


class SectionRecord:
    def __init__(self, block: BlockRecord, section_num: int, tile_grid_num: int):
        self.logger = Logger("Section Record")
        self.block = block
        self.section_num = section_num
        self.tile_grid_num = tile_grid_num

        self.section_id = tuple([self.section_num, self.tile_grid_num])

        self.tile_map = {}

        self.tile_id_map = None

        if self.block is not None:
            self.block.register_section(self)

    def register_tile(self, tile: TileRecord):
        self.tile_map[tile.tile_id] = tile

    def get_tile(self, tile_id: int):
        if tile_id in self.tile_map.keys():
            return self.tile_map[tile_id]
        else:
            return None

    def compute_tile_id_map(self):
        tile_to_coords = {}
        for t in self.tile_map.values():
            tile_to_coords[t.tile_id] = tuple([t.x, t.y])

        xx, yy = set(), set()
        for t in tile_to_coords.values():
            xx.add(t[0])
            yy.add(t[1])

        xx = list(sorted(xx))
        yy = list(sorted(yy))

        self.tile_id_map = np.zeros((len(yy), len(xx)), dtype=int) - 1
        for t, (x, y) in tile_to_coords.items():
            self.tile_id_map[yy.index(y), xx.index(x)] = t

    def get_name(self):
        return "s" + str(self.section_id[0]) + "_g" + str(self.section_id[1])

    def save(self, path):
        if exists(path):
            self.logger.warning("Section already exists.")
        else:
            mkdir(path)
            tile_id_map_path = self.get_name() + "_tile_id_map.npz"

            tiles = {}
            for tile_id, tile in self.tile_map.items():
                tiles[tile_id] = tile.get_tile_dict()

            section_dict = {
                "section_num": self.section_num,
                "tile_grid_num": self.tile_grid_num,
                "section_id": self.section_id,
                "n_tiles": len(self.tile_map),
                "tile_map": tiles,
                "tile_id_map": join(".", tile_id_map_path),
            }
            with open(join(path, "section.json"), "w") as f:
                json.dump(section_dict, f, indent=4)

            if self.tile_id_map is not None:
                np.savez(join(path, tile_id_map_path), tile_id_map=self.tile_id_map)

    def load(self, path):
        path_ = join(path, "section.json")
        if not exists(path_):
            self.logger.warning(f"Section not found: {path_}")
        else:
            with open(path_) as f:
                section_dict = json.load(f)

            assert (
                self.section_num == section_dict["section_num"]
            ), "{}[section_num] incompatible with directory tree."
            assert (
                self.tile_grid_num == section_dict["tile_grid_num"]
            ), "{}[tile_grid_num] incompatible with directory tree."

            for tile_id, tile_dict in section_dict["tile_map"].items():
                t = TileRecord(
                    section=self,
                    path=tile_dict["path"],
                    tile_id=tile_dict["tile_id"],
                    x=tile_dict["x"],
                    y=tile_dict["y"],
                    resolution_xy=tile_dict["resolution_xy"],
                )
                self.register_tile(t)

            tile_id_map_path = join(path, self.get_name() + "_tile_id_map.npz")
            if exists(tile_id_map_path):
                data = np.load(tile_id_map_path)
                if "tile_id_map" in data.files:
                    self.tile_id_map = data["tile_id_map"]
