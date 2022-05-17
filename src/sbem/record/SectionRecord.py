from __future__ import annotations

import json
from os import mkdir
from os.path import exists, join
from typing import TYPE_CHECKING

import numpy as np
from numcodecs import Blosc

from sbem.record.TileRecord import TileRecord

if TYPE_CHECKING:
    from sbem.record import BlockRecord


class SectionRecord:
    """
    A section in an SBEM experiment belongs to a block and contains many tiles.
    """

    def __init__(
        self,
        block: BlockRecord,
        section_num: int,
        tile_grid_num: int,
        save_dir: str = None,
        logger=None,
    ):
        """
        :param block: to which this section belongs.
        :param section_num: Number of this section.
        :param tile_grid_num: Tile grid number of this section.
        :param save_dir: Directory to which this section is saved.
        """
        self.logger = logger
        self.block = block
        self.section_num = section_num
        self.tile_grid_num = tile_grid_num

        self.section_id = tuple([self.section_num, self.tile_grid_num])
        if self.get_name() in self.block.zarr_block.keys():
            self.zarr_section = self.block.zarr_block[self.get_name()]
        else:
            self.zarr_section = self.block.zarr_block.create_group(
                self.get_name(), overwrite=False
            )

        if save_dir is not None:
            self.save_dir = join(save_dir, self.get_name())
        else:
            self.save_dir = None

        self.tile_map = {}

        self.tile_id_map = None

        if self.block is not None:
            self.block.add_section(self)

    def add_tile(self, tile: TileRecord):
        """
        Add a tile to this section.

        :param tile: to register.
        """
        self.tile_map[tile.tile_id] = tile

    def get_tile(self, tile_id: int):
        """
        Get tile registered with this section.

        :param tile_id:
        :return: `TileRecord` or `None` if tile does not exist.
        """
        if tile_id in self.tile_map.keys():
            return self.tile_map[tile_id]
        else:
            return None

    def get_tile_data_map(self):
        """
        Get a tile-data-map mapping tile (x, y) coordinates to the loaded
        image data.

        :return: tile-data-map
        """
        tile_data_map = {}
        for y in range(self.tile_id_map.shape[0]):
            for x in range(self.tile_id_map.shape[1]):
                if self.tile_id_map[y, x] != -1:
                    tile_data_map[(x, y)] = self.tile_map[
                        self.tile_id_map[y, x]
                    ].get_tile_data()
        return tile_data_map

    def compute_tile_id_map(self):
        """
        Computes te tile-id-map mapping (y, x) coordinates to a tile.

        :return: tile-id-map
        """
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
        """
        :return: section name i.e. 's{section_number}_g{grid_number}'
        """
        return "s" + str(self.section_id[0]) + "_g" + str(self.section_id[1])

    def write_stitched(self, stitched, mask):
        zarr_mask = self.zarr_section.create(
            f"mask_grid-{self.section_id[1]}",
            shape=mask.shape,
            chunks=4096,
            dtype=mask.dtype,
            compressor=Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE),
        )
        zarr_mask[...] = mask[...]
        zarr_stitched = self.zarr_section.create(
            f"stitched_grid-{self.section_id[1]}",
            shape=stitched.shape,
            chunks=4096,
            dtype=stitched.dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        )
        zarr_stitched[...] = stitched[...]

    def save(self):
        """
        Save section to `save_dir`.
        """
        assert self.save_dir is not None, "Save dir not set."
        if not exists(self.save_dir):
            mkdir(self.save_dir)
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
            "tile_id_map": join("", tile_id_map_path),
        }
        with open(join(self.save_dir, "section.json"), "w") as f:
            json.dump(section_dict, f, indent=4)

        if self.tile_id_map is not None:
            np.savez(
                join(self.save_dir, tile_id_map_path), tile_id_map=self.tile_id_map
            )

    def load(self, path):
        """
        Load section from disk.

        :param path: to section directory.
        """
        path_ = join(path, "section.json")
        if not exists(path_) and self.logger is not None:
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
                    logger=self.logger,
                )
                self.add_tile(t)

            tile_id_map_path = join(path, self.get_name() + "_tile_id_map.npz")
            if exists(tile_id_map_path):
                data = np.load(tile_id_map_path)
                if "tile_id_map" in data.files:
                    self.tile_id_map = data["tile_id_map"]
