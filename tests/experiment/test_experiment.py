import configparser
from os import makedirs
from os.path import join

from sbem.experiment.Experiment import Experiment


def test_experiment(tmpdir):
    logs_path = join(tmpdir, "sbem", "bloc", "meta", "logs")
    makedirs(logs_path)

    config = configparser.ConfigParser()
    config["grids"] = {"grid_active": "[0, 1, 0]", "pixel_size": "[11.0, 11.0, 11.0]"}

    with open(join(logs_path, "config_example.txt"), "w") as f:
        config.write(f)

    metadata_path = join(logs_path, "metadata_example.txt")
    with open(metadata_path, "w") as f:
        f.write(
            "TILE: {'tileid': '0001.0431.05283', 'timestamp': "
            "1628277040, 'filename': "
            "'tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif', "
            "'tile_width': 3072, 'tile_height': 2304, "
            "'wd_stig_xy': [0.006170215, -0.9843520000000001, 0.6759430000000001], "
            "'glob_x': -450885, 'glob_y': -744566, 'glob_z': 132025, "
            "'slice_counter': 5283}\n"
        )

    exp = Experiment("name", tmpdir)
    exp.add_block(join(tmpdir, "sbem", "bloc"), "bloc", "g0001", 11)
    assert len(exp.blocks) == 1
    assert len(exp.blocks["bloc"].sections) == 1
    assert len(exp.blocks["bloc"].sections[(5283, 1)].tile_map) == 1

    exp.save()
    exp_load = Experiment()
    exp_load.load(join(tmpdir, "name"))
    assert len(exp_load.blocks) == 1
    assert len(exp_load.blocks["bloc"].sections) == 1
    assert len(exp_load.blocks["bloc"].sections[(5283, 1)].tile_map) == 1
