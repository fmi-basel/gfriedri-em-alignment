from __future__ import annotations

from os.path import exists
from typing import TYPE_CHECKING

from skimage.io import imread

if TYPE_CHECKING:
    from record.SectionRecord import SectionRecord


class TileRecord:
    """
    A tile belongs to a section, which belongs to a block, which belongs to
    a SBEM experiment.
    """

    def __init__(
        self,
        section: SectionRecord,
        path: str,
        tile_id: int,
        x: float,
        y: float,
        resolution_xy: float,
        logger=None,
    ):
        """
        :param section: to which this tile belongs.
        :param path: to the tile image file.
        :param tile_id: of the tile.
        :param x: global pixel-coordinate of the tile.
        :param y: global pixel-coordinate of the tile.
        :param resolution_xy: of the image data.
        """
        self.logger = logger
        self.section = section

        if not exists(path) and self.logger is not None:
            self.logger.warning(f"{path} does not exist.")
        self.path = path

        self.tile_id = tile_id
        self.x = x
        self.y = y
        self.resolution_xy = resolution_xy

        if self.section is not None:
            self.section.add_tile(self)

    def get_tile_data(self):
        """
        Load tile image data.

        :return: tile data
        """
        return imread(self.path)

    def get_tile_dict(self):
        """
        Dict summary of this tile.

        :return: dict
        """
        tile_dict = {
            "path": self.path,
            "tile_id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "resolution_xy": self.resolution_xy,
        }
        return tile_dict
