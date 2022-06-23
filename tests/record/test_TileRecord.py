import os
import shutil
import tempfile
from os.path import join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from tifffile import imsave

from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from src.sbem.experiment import Experiment
from src.sbem.record.BlockRecord import BlockRecord


class TileTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_tile_record(self):
        exp_dir = join(self.tmp_dir, "exp")
        os.mkdir(exp_dir)
        exp = Experiment("test", exp_dir)
        block = BlockRecord(exp, None, "bloc1", None)

        img = np.random.rand(127, 193)
        tile_id = 10
        img_path = join(self.tmp_dir, f"tile_{tile_id}.tif")
        imsave(img_path, img)

        section = SectionRecord(
            block=block,
            section_num=1,
            tile_grid_num=1,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )
        tile = TileRecord(
            section=section,
            path=img_path,
            tile_id=tile_id,
            x=34,
            y=94,
            resolution_xy=2.5,
        )

        tile_data = tile.get_tile_data()
        assert_array_equal(img, tile_data)

        tile_dict = tile.get_tile_dict()
        assert tile_dict["path"] == img_path
        assert tile_dict["tile_id"] == tile_id
        assert tile_dict["x"] == 34
        assert tile_dict["y"] == 94
        assert tile_dict["resolution_xy"] == 2.5
