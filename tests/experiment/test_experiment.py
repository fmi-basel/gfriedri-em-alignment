import shutil
import tempfile
from os import makedirs
from os.path import join
from unittest import TestCase

from sbem.experiment import Experiment


class ExperimentTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        logs_path = join(self.tmp_dir, "sbem", "bloc", "meta", "logs")
        makedirs(logs_path)
        metadata_path = join(logs_path, "metadata_example.txt")
        with open(metadata_path, "w") as f:
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

    def test_experiment(self):
        exp = Experiment("name", self.tmp_dir)
        exp.parse_block(
            join(self.tmp_dir, "sbem", "bloc"),
            "bloc",
            "g0001",
            11,
            tile_width=3072,
            tile_height=2304,
            tile_overlap=200,
        )
        assert len(exp.blocks) == 1
        assert len(exp.blocks["bloc"].sections) == 1
        assert len(exp.blocks["bloc"].sections[(5283, 1)].tile_map) == 1

        exp.save()
        exp_load = Experiment()
        exp_load.load(join(self.tmp_dir, "name"))
        assert len(exp_load.blocks) == 1
        assert len(exp_load.blocks["bloc"].sections) == 1
        assert len(exp_load.blocks["bloc"].sections[(5283, 1)].tile_map) == 1
