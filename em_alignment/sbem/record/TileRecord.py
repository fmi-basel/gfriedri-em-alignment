from __future__ import annotations

from logging import Logger
from os.path import exists
from typing import TYPE_CHECKING

from skimage.io import imread

if TYPE_CHECKING:
    from sbem.record.SectionRecord import SectionRecord


class TileRecord:
    def __init__(
        self,
        section: SectionRecord,
        path: str,
        tile_id: int,
        x: float,
        y: float,
        resolution_xy: float,
    ):
        self.logger = Logger("Tile Record")
        self.section = section

        if not exists(path):
            self.logger.warning(f"{path} does not exist.")
        self.path = path

        self.tile_id = tile_id
        self.x = x
        self.y = y
        self.resolution_xy = resolution_xy

        if self.section is not None:
            self.section.register_tile(self)

    def get_tile_data(self):
        return imread(self.path)

    def get_tile_dict(self):
        tile_dict = {
            "path": self.path,
            "tile_id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "resolution_xy": self.resolution_xy,
        }
        return tile_dict
