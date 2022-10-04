import json
import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from ruyaml import YAML
from tifffile import imsave

from sbem.record_v2.Section import Section
from sbem.record_v2.Tile import Tile


class SectionTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.img = np.random.randint(0, 3200, size=(1320, 3021), dtype=np.uint16)

        imsave(join(self.tmp_dir, "img.tif"), self.img)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_section_arg_order(self):
        name = "section_init"
        stitched = (False,)
        skip = True
        acquisition = "run_0"
        sample = None
        section_num = 123
        tile_grid_num = 1
        thickness = 11.1
        tile_height = 3420
        tile_width = 4200
        tile_overlap = 200
        license = "Fake license."
        sec = Section(
            name,
            stitched,
            skip,
            acquisition,
            sample,
            section_num,
            tile_grid_num,
            thickness,
            tile_height,
            tile_width,
            tile_overlap,
            license,
        )

        assert sec.get_name() == name
        assert sec.get_license() == license
        assert sec.get_sample() == sample
        assert sec.get_section_num() == section_num
        assert sec.get_tile_grid_num() == tile_grid_num
        assert sec.get_acquisition() == acquisition
        assert sec.get_thickness() == thickness
        assert sec.get_tile_height() == tile_height
        assert sec.get_tile_width() == tile_width
        assert sec.get_tile_overlap() == tile_overlap
        assert sec.is_stitched() == stitched
        assert sec.skip() == skip
        assert sec.get_tile_id_map() is None
        assert len(sec.tiles) == 0
        assert sec._fully_initialized
        assert sec.get_section_id() == f"s{section_num}_g{tile_grid_num}"

    def test_partial_init(self):
        name = "section_init"
        stitched = False
        skip = True
        acquisition = "run_0"
        sample = None
        section_num = 123
        tile_grid_num = 1
        thickness = 11.1
        tile_height = 3420
        tile_width = 4200
        tile_overlap = 200
        license = "Fake license."
        sec = Section.lazy_loading(
            name, stitched, skip, acquisition, "./section_00/sample.yaml"
        )

        assert not sec._fully_initialized
        assert sec.get_name() == name
        assert sec.is_stitched() == stitched
        assert sec.skip() == skip
        assert sec.get_acquisition() == acquisition
        self.assertRaises(RuntimeError, sec.add_tile)
        self.assertRaises(RuntimeError, sec.get_tile)
        self.assertRaises(RuntimeError, sec.get_section_num)
        self.assertRaises(RuntimeError, sec.get_tile_grid_num)
        self.assertRaises(RuntimeError, sec.get_thickness)
        self.assertRaises(RuntimeError, sec.get_tile_height)
        self.assertRaises(RuntimeError, sec.get_tile_width)
        self.assertRaises(RuntimeError, sec.get_tile_overlap)
        self.assertRaises(RuntimeError, sec._compute_tile_id_map)
        self.assertRaises(RuntimeError, sec.get_tile_id_map)
        self.assertRaises(RuntimeError, sec.to_dict)
        self.assertRaises(RuntimeError, sec.get_section_id)
        self.assertRaises(RuntimeError, sec.save)

        tile = Tile(
            section=None,
            tile_id=1,
            path="/fake.tif",
            stage_x=0,
            stage_y=0,
            resolution_xy=11.0,
        )
        details = {
            "format_version": "0.1.0",
            "license": license,
            "section_num": section_num,
            "tile_grid_num": tile_grid_num,
            "thickness": thickness,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "tile_overlap": tile_overlap,
            "tiles": [tile.to_dict()],
        }

        sec._load_details(details)
        assert sec.get_name() == name
        assert sec.get_license() == license
        assert sec.get_sample() == sample
        assert sec.get_section_num() == section_num
        assert sec.get_tile_grid_num() == tile_grid_num
        assert sec.get_acquisition() == acquisition
        assert sec.get_thickness() == thickness
        assert sec.get_tile_height() == tile_height
        assert sec.get_tile_width() == tile_width
        assert sec.get_tile_overlap() == tile_overlap
        assert sec.is_stitched() == stitched
        assert sec.skip() == skip
        assert sec.get_tile_id_map() == np.array([[1]])
        assert len(sec.tiles) == 1
        assert sec._fully_initialized

    def test_save(self):
        name = "section_init"
        stitched = False
        skip = True
        acquisition = "run_0"
        sample = None
        section_num = 123
        tile_grid_num = 1
        thickness = 11.1
        tile_height = 3420
        tile_width = 4200
        tile_overlap = 200
        license = "Fake license."
        sec = Section(
            name,
            stitched,
            skip,
            acquisition,
            sample,
            section_num,
            tile_grid_num,
            thickness,
            tile_height,
            tile_width,
            tile_overlap,
            license,
        )

        sec.save(path=self.tmp_dir, overwrite=False)
        assert exists(join(self.tmp_dir, sec.get_section_id(), "section.yaml"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, sec.get_section_id(), "section.yaml")) as f:
            dict = yaml.load(f)

        assert dict["license"] == license
        assert dict["format_version"] == "0.1.0"
        assert dict["section_num"] == section_num
        assert dict["tile_grid_num"] == tile_grid_num
        assert dict["thickness"] == thickness
        assert dict["tile_height"] == tile_height
        assert dict["tile_width"] == tile_width
        assert dict["tile_overlap"] == tile_overlap
        assert dict["tiles"] == []

        # details in dedicated section.yaml
        sec_loaded = Section.lazy_loading(
            name=name,
            stitched=stitched,
            skip=skip,
            acquisition=acquisition,
            details=join(".", sec.get_section_id(), "section.yaml"),
        )
        sec_loaded.load_from_yaml(
            join(self.tmp_dir, sec.get_section_id(), "section.yaml")
        )
        assert sec_loaded.get_name() == name
        assert sec_loaded.get_license() == license
        assert sec_loaded.get_sample() == sample
        assert sec_loaded.get_section_num() == section_num
        assert sec_loaded.get_tile_grid_num() == tile_grid_num
        assert sec_loaded.get_acquisition() == acquisition
        assert sec_loaded.get_thickness() == thickness
        assert sec_loaded.get_tile_height() == tile_height
        assert sec_loaded.get_tile_width() == tile_width
        assert sec_loaded.get_tile_overlap() == tile_overlap
        assert sec_loaded.is_stitched() == stitched
        assert sec_loaded.skip() == skip
        assert sec_loaded.get_tile_id_map() is None
        assert len(sec_loaded.tiles) == 0
        assert sec_loaded._fully_initialized

        # section details provided as dict.
        sec_loaded = Section.lazy_loading(
            name=name,
            stitched=stitched,
            skip=skip,
            acquisition=acquisition,
            details=dict,
        )
        assert sec_loaded.get_name() == name
        assert sec_loaded.get_license() == license
        assert sec_loaded.get_sample() == sample
        assert sec_loaded.get_section_num() == section_num
        assert sec_loaded.get_tile_grid_num() == tile_grid_num
        assert sec_loaded.get_acquisition() == acquisition
        assert sec_loaded.get_thickness() == thickness
        assert sec_loaded.get_tile_height() == tile_height
        assert sec_loaded.get_tile_width() == tile_width
        assert sec_loaded.get_tile_overlap() == tile_overlap
        assert sec_loaded.is_stitched() == stitched
        assert sec_loaded.skip() == skip
        assert sec_loaded.get_tile_id_map() is None
        assert len(sec_loaded.tiles) == 0
        assert sec_loaded._fully_initialized

        tile = Tile(
            section=sec,
            tile_id=1,
            path="/fake.tif",
            stage_x=0,
            stage_y=0,
            resolution_xy=11.0,
        )

        dict = sec.to_dict()
        assert len(dict["tiles"]) == 1
        assert dict["tiles"][0] == tile.to_dict()

        self.assertRaises(FileExistsError, sec.save, path=self.tmp_dir, overwrite=False)

        sec.save(path=self.tmp_dir, overwrite=True)
        sec_loaded.load_from_yaml(
            join(self.tmp_dir, sec.get_section_id(), "section.yaml")
        )

        assert len(sec_loaded.tiles) == 1
        assert sec_loaded.get_tile(1).get_section() == sec_loaded
        assert sec_loaded.get_tile(1).get_tile_path() == "/fake.tif"

    def test_add_tile(self):
        sec = Section(
            "section_init", False, True, "run_0", None, 123, 1, 11.1, 3420, 4200, 200
        )

        tile = Tile(sec, 2, "/fake_path.tif", 1, 1, 11.0)
        assert tile.get_section() == sec

        tile_1 = Tile(None, 3, "/another_fake.tif", 2, 1, 11.1)
        sec.add_tile(tile_1)
        assert tile_1.get_section() == sec

        assert len(sec.tiles) == 2

        assert sec.get_tile(2) == tile
        assert sec.get_tile(3) == tile_1

    def test_tile_id_map(self):
        sec = Section(
            "section_init", False, True, "run_0", None, 123, 1, 11.1, 3072, 2304, 200
        )

        Tile(
            sec,
            path="/not/important.tif",
            tile_id=3,
            stage_x=0,
            stage_y=0,
            resolution_xy=1.2,
        )
        Tile(
            sec,
            path="/not/important.tif",
            tile_id=4,
            stage_x=2104,
            stage_y=0,
            resolution_xy=1.2,
        )
        Tile(
            sec,
            path="/not/important.tif",
            tile_id=5,
            stage_x=0,
            stage_y=2872,
            resolution_xy=1.2,
        )
        Tile(
            sec,
            path="/not/important.tif",
            tile_id=6,
            stage_x=2104,
            stage_y=2872,
            resolution_xy=1.2,
        )

        tile_id_map = sec.get_tile_id_map()
        assert tile_id_map[0, 0] == 3
        assert tile_id_map[0, 1] == 4
        assert tile_id_map[1, 0] == 5
        assert tile_id_map[1, 1] == 6

        tile_id_path = join(self.tmp_dir, "tile_id_map.json")
        tile_id_map = sec.get_tile_id_map(path=tile_id_path)

        assert exists(tile_id_path)

        with open(tile_id_path) as f:
            loaded_tile_id_map = np.array(json.load(f))

        assert_array_equal(loaded_tile_id_map, tile_id_map)

        loaded_tile_id_map[0, 0] = -1
        with open(tile_id_path, "w") as f:
            json.dump(loaded_tile_id_map.tolist(), f)

        # Load from disk
        tile_id_map = sec.get_tile_id_map(path=tile_id_path)

        assert_array_equal(loaded_tile_id_map, tile_id_map)

        # Recompute and ignore the one on disk
        tile_id_map = sec.get_tile_id_map()
        assert tile_id_map[0, 0] == 3
        assert tile_id_map[0, 1] == 4
        assert tile_id_map[1, 0] == 5
        assert tile_id_map[1, 1] == 6
