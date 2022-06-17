import os
import shutil
import tempfile
from os.path import join
from unittest import TestCase

import numpy as np
from tifffile import imsave

from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from src.sbem.experiment import Experiment


class SectionTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_section_record(self):
        exp_dir = join(self.tmp_dir, "exp")
        os.mkdir(exp_dir)
        exp = Experiment("test", exp_dir)
        block = BlockRecord(exp, None, "bloc1", None)

        section_num = 745
        tile_grid_num = 1
        section = SectionRecord(
            block=block,
            section_num=section_num,
            tile_grid_num=tile_grid_num,
            save_dir=self.tmp_dir,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )

        data = np.random.rand(93, 84)
        data_path = join(self.tmp_dir, "data.tif")
        imsave(data_path, data)

        # tile-map
        #  3, 4,
        # 5,  6
        tile_03 = TileRecord(
            section, path=data_path, tile_id=3, x=0, y=0, resolution_xy=1.2
        )
        section.add_tile(tile_03)
        tile_04 = TileRecord(
            section, path=data_path, tile_id=4, x=2104, y=0, resolution_xy=1.2
        )
        section.add_tile(tile_04)
        tile_05 = TileRecord(
            section, path=data_path, tile_id=5, x=0, y=2872, resolution_xy=1.2
        )
        section.add_tile(tile_05)
        tile_06 = TileRecord(
            section, path=data_path, tile_id=6, x=2104, y=2872, resolution_xy=1.2
        )
        section.add_tile(tile_06)

        assert section.get_tile(3) == tile_03
        assert section.get_tile(4) == tile_04
        assert section.get_tile(5) == tile_05
        assert section.get_tile(6) == tile_06

        section.compute_tile_id_map()
        assert section.tile_id_map[0, 0] == 3
        assert section.tile_id_map[0, 1] == 4
        assert section.tile_id_map[1, 0] == 5
        assert section.tile_id_map[1, 1] == 6

        assert section.get_name() == f"s{section_num}_g{tile_grid_num}"

        section.save()
        load_section = SectionRecord(
            block=block,
            section_num=section_num,
            tile_grid_num=tile_grid_num,
            save_dir=None,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )
        load_section.load(join(self.tmp_dir, load_section.get_name()))
        assert section.section_num == load_section.section_num
        assert len(load_section.tile_map) == 4
        assert section.get_tile(3).tile_id == 3
        assert section.get_tile(4).tile_id == 4
        assert section.get_tile(5).tile_id == 5
        assert section.get_tile(6).tile_id == 6

        assert section.get_tile(3).x == 0
        assert section.get_tile(3).y == 0
        assert section.get_tile(3).resolution_xy == 1.2

        assert section.tile_id_map[0, 0] == 3
        assert section.tile_id_map[0, 1] == 4
        assert section.tile_id_map[1, 0] == 5
        assert section.tile_id_map[1, 1] == 6

    def test_tile_id_with_gaps(self):
        exp_dir = join(self.tmp_dir, "exp")
        os.mkdir(exp_dir)
        exp = Experiment("test", exp_dir)
        block = BlockRecord(exp, None, "bloc1", None)

        section_num = 745
        tile_grid_num = 1
        section = SectionRecord(
            block=block,
            section_num=section_num,
            tile_grid_num=tile_grid_num,
            save_dir=self.tmp_dir,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )

        data = np.random.rand(93, 84)
        data_path = join(self.tmp_dir, "data.tif")
        imsave(data_path, data)

        # tile-map
        #  3, 4, -1
        # -1, -1, -1
        # 8, -1, 9
        tile_03 = TileRecord(
            section, path=data_path, tile_id=3, x=0, y=0, resolution_xy=1.2
        )
        section.add_tile(tile_03)
        tile_04 = TileRecord(
            section, path=data_path, tile_id=4, x=2104, y=0, resolution_xy=1.2
        )
        section.add_tile(tile_04)
        tile_08 = TileRecord(
            section, path=data_path, tile_id=8, x=0, y=2872 * 2, resolution_xy=1.2
        )
        section.add_tile(tile_08)
        tile_09 = TileRecord(
            section,
            path=data_path,
            tile_id=9,
            x=2104 * 2,
            y=2872 * 2,
            resolution_xy=1.2,
        )
        section.add_tile(tile_09)

        assert section.get_tile(3) == tile_03
        assert section.get_tile(4) == tile_04
        assert section.get_tile(8) == tile_08
        assert section.get_tile(9) == tile_09

        section.compute_tile_id_map()
        assert section.tile_id_map[0, 0] == 3
        assert section.tile_id_map[0, 1] == 4
        assert section.tile_id_map[0, 2] == -1
        assert section.tile_id_map[1, 0] == -1
        assert section.tile_id_map[1, 1] == -1
        assert section.tile_id_map[1, 2] == -1
        assert section.tile_id_map[2, 0] == 8
        assert section.tile_id_map[2, 1] == -1
        assert section.tile_id_map[2, 2] == 9

        assert section.get_name() == f"s{section_num}_g{tile_grid_num}"

        section.save()
        load_section = SectionRecord(
            block=block,
            section_num=section_num,
            tile_grid_num=tile_grid_num,
            save_dir=None,
        )
        load_section.load(join(self.tmp_dir, load_section.get_name()))
        assert section.section_num == load_section.section_num
        assert len(load_section.tile_map) == 4
        assert section.get_tile(3).tile_id == 3
        assert section.get_tile(4).tile_id == 4
        assert section.get_tile(8).tile_id == 8
        assert section.get_tile(9).tile_id == 9

        assert section.get_tile(3).x == 0
        assert section.get_tile(3).y == 0
        assert section.get_tile(3).resolution_xy == 1.2

        assert section.tile_id_map[0, 0] == 3
        assert section.tile_id_map[0, 1] == 4
        assert section.tile_id_map[0, 2] == -1
        assert section.tile_id_map[1, 0] == -1
        assert section.tile_id_map[1, 1] == -1
        assert section.tile_id_map[1, 2] == -1
        assert section.tile_id_map[2, 0] == 8
        assert section.tile_id_map[2, 1] == -1
        assert section.tile_id_map[2, 2] == 9
