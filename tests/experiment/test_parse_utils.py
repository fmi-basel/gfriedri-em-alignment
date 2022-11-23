import configparser
import shutil
import tempfile
from os import makedirs
from os.path import join
from unittest import TestCase

from sbem.experiment import Experiment
from sbem.record.Sample import Sample
from src.sbem.experiment.parse_utils import (
    get_acquisition_config,
    get_tile_metadata,
    get_tile_spec_from_SBEMtile,
    parse_and_add_sections,
    read_tile_metadata,
)


class ParseUtilsTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        config = configparser.ConfigParser()
        config["grids"] = {
            "grid_active": "[0, 1, 0]",
            "pixel_size": "[11.0, 11.0, 11.0]",
        }

        self.metadata_path = join(self.tmp_dir, "meta", "logs", "metadata_.txt")
        makedirs(join(self.tmp_dir, "meta", "logs"))
        with open(self.metadata_path, "w") as f:
            f.write(
                "SESSION: {'timestamp': 1628084775, 'eht': 1.8, "
                "'beam_current': 1100, 'wd_stig_xy_default': "
                "[0.006163975223898888, -0.9843519926071167, "
                "0.5319430232048035], 'slice_thickness': 25, "
                "'grids': ['0000', '0001', '0002'], "
                "'grid_origins': [[-2.478, -719.711], [56.818, -733.46], "
                "[33.541, -713.294]], 'rotation_angles': [0.0, 0.0, 0.0], "
                "'pixel_sizes': [11.0, 11.0, 11.0], "
                "'dwell_times': [0.2, 0.2, 0.2], "
                "'contrast': 5.57, 'brightness': 11.17, "
                "'email_addresses: ': ['email@doma.in']}\n"
            )
            f.write(
                "TILE: {'tileid': '0001.0431.05283', 'timestamp': "
                "1628277040, 'filename': "
                "'tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif', "
                "'tile_width': 3072, 'tile_height': 2304, "
                "'wd_stig_xy': [0.006170215, -0.9843520000000001, 0.6759430000000001], "
                "'glob_x': -450885, 'glob_y': -744566, 'glob_z': 132025, "
                "'slice_counter': 5283}\n"
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_get_acquistion_config(self):
        result = get_acquisition_config(self.metadata_path)
        assert result["timestamp"] == 1628084775
        assert result["eht"] == 1.8
        assert result["beam_current"] == 1100
        assert result["wd_stig_xy_default"] == [
            0.006163975223898888,
            -0.9843519926071167,
            0.5319430232048035,
        ]
        assert result["slice_thickness"] == 25
        assert result["grids"] == ["0000", "0001", "0002"]
        assert result["grid_origins"] == [
            [-2.478, -719.711],
            [56.818, -733.46],
            [33.541, -713.294],
        ]
        assert result["rotation_angles"] == [0.0, 0.0, 0.0]
        assert result["pixel_sizes"] == [11.0, 11.0, 11.0]
        assert result["dwell_times"] == [0.2, 0.2, 0.2]
        assert result["contrast"] == 5.57
        assert result["brightness"] == 11.17
        assert result["email_addresses: "] == ["email@doma.in"]

    def test_get_tile_spec_from_SBEMtile(self):
        exp_path = "/tmp/test"
        file_name = (
            "tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif"
        )
        tile = {
            "tileid": "0001.0431.05283",
            "timestamp": "1628277040",
            "filename": file_name,
            "tile_width": "3072",
            "tile_height": "2304",
            "wd_stig_xy": "[0.006170215, -0.9843520000000001, 0.6759430000000001]",
            "glob_x": "-450885",
            "glob_y": "-744566",
            "glob_z": "132025",
            "slice_counter": "5283",
        }
        resolution_xy = 11.0

        result = get_tile_spec_from_SBEMtile(exp_path, tile, resolution_xy)

        assert result["tile_id"] == 431
        assert result["tile_file"] == join(exp_path, file_name)
        assert result["x"] == float(-450885) // resolution_xy
        assert result["y"] == float(-744566) // resolution_xy
        assert result["z"] == 5283

    def test_read_tile_metadata(self):
        exp_path = "/tmp/experiment"
        resolution_xy = 11.0

        result = read_tile_metadata(exp_path, self.metadata_path, 1, resolution_xy)

        assert len(result) == 1
        assert result[0]["tile_id"] == 431
        assert result[0]["tile_file"] == join(
            exp_path,
            "tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif",
        )
        assert result[0]["x"] == -450885 // resolution_xy
        assert result[0]["y"] == -744566 // resolution_xy
        assert result[0]["z"] == 5283

    def test_get_tile_metadata(self):
        tile_specs = get_tile_metadata("/tmp/experiment", [self.metadata_path], 1, 11.0)
        assert len(tile_specs) == 1

    def test_parse_and_add_sections(self):
        exp = Experiment(
            "name", "desc", "docu", root_dir=self.tmp_dir, authors=[], exist_ok=True
        )
        sample = Sample(
            experiment=exp,
            name="Sample",
            description="desc",
            documentation="",
            aligned_data="",
        )

        parse_and_add_sections(
            sbem_root_dir=self.tmp_dir,
            sample=sample,
            acquisition="run_0",
            tile_grid="g0001",
            thickness=25.0,
            resolution_xy=11.0,
            tile_width=3072,
            tile_height=2304,
            tile_overlap=200,
        )

        sec = sample.get_section("s5283_g1")
        assert sec.get_section_num() == 5283
        assert sec.get_tile_overlap() == 200
        assert sec.get_tile_width() == 3072
        assert sec.get_tile_height() == 2304
        assert not sec.is_stitched()
        assert not sec.skip()
        sec.set_alignment_mesh("/a/path/mesh.npz")
        assert sec.get_alignment_mesh() == "/a/path/mesh.npz"
        assert len(sec.tiles) == 1

        tile = sec.get_tile(431)
        assert tile.x == -450885 // 11.0
        assert tile.y == -744566 // 11.0
