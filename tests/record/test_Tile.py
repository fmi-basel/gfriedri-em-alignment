import shutil
import tempfile
from os.path import join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from tifffile import imsave

from sbem.record.Tile import Tile


class TileTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.img = np.random.randint(0, 3200, size=(1320, 3021), dtype=np.uint16)

        imsave(join(self.tmp_dir, "img.tif"), self.img)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_tile(self):
        section = None
        tile_id = 42
        path = join(self.tmp_dir, "img.tif")
        stage_x = -32.1
        stage_y = 89.3
        resolution_xy = 11.8
        unit = "um"

        tile = Tile(section, tile_id, path, stage_x, stage_y, resolution_xy, unit)

        assert tile.get_tile_id() == tile_id
        assert_array_equal(tile.get_tile_data(), self.img)
        assert tile.get_tile_path() == path
        assert tile.get_resolution() == resolution_xy
        assert tile.get_unit() == unit
        assert tile.get_section() == section
        assert tile.x == stage_x
        assert tile.y == stage_y

        tile_dict = tile.to_dict()
        assert tile_dict["tile_id"] == tile_id
        assert tile_dict["path"] == path
        assert tile_dict["stage_x"] == stage_x
        assert tile_dict["stage_y"] == stage_y
        assert tile_dict["resolution_xy"] == resolution_xy
        assert tile_dict["unit"] == unit

        tile_2 = Tile.from_dict(tile_dict)
        assert tile != tile_2
        assert tile.get_tile_id() == tile_id
        assert_array_equal(tile.get_tile_data(), self.img)
        assert tile.get_tile_path() == path
        assert tile.get_resolution() == resolution_xy
        assert tile.get_unit() == unit
        assert tile.get_section() == section
        assert tile.x == stage_x
        assert tile.y == stage_y
