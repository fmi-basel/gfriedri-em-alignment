import configparser
from os.path import join

from sbem.experiment import (
    get_acquisition_config,
    get_tile_metadata,
    get_tile_spec_from_SBEMtile,
    read_tile_metadata,
)


def test_get_acquistion_config(tmpdir):
    config = configparser.ConfigParser()
    config["grids"] = {"grid_active": "[0, 1, 0]", "pixel_size": "[11.0, 11.0, 11.0]"}

    with open(join(tmpdir, "an_example_config.txt"), "w") as f:
        config.write(f)

    result = get_acquisition_config(join(tmpdir, "an_example_metadata.txt"))
    assert "grids" in result.sections()
    assert result["grids"]["grid_active"] == "[0, 1, 0]"
    assert result["grids"]["pixel_size"] == "[11.0, 11.0, 11.0]"


def test_get_tile_spec_from_SBEMtile():
    exp_path = "/tmp/test"
    file_name = "tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif"
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


def test_read_tile_metadata(tmpdir):
    metadata_path = join(tmpdir, "test_metadata.txt")
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

    exp_path = "/tmp/experiment"
    resolution_xy = 11.0

    result = read_tile_metadata(exp_path, metadata_path, resolution_xy)

    assert len(result) == 1
    assert result[0]["tile_id"] == 431
    assert result[0]["tile_file"] == join(
        exp_path, "tiles/g0001/t0431/20210630_Dp_190326Bb_run04_g0001_t0431_s05283.tif"
    )
    assert result[0]["x"] == -450885 // resolution_xy
    assert result[0]["y"] == -744566 // resolution_xy
    assert result[0]["z"] == 5283


def test_get_tile_metadata(tmpdir):
    config = configparser.ConfigParser()
    config["grids"] = {"grid_active": "[0, 1, 0]", "pixel_size": "[11.0, 11.0, 11.0]"}

    with open(join(tmpdir, "test_config.txt"), "w") as f:
        config.write(f)

    metadata_path = join(tmpdir, "test_metadata.txt")
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

    tile_specs = get_tile_metadata(
        "/tmp/experiment", [join(tmpdir, "test_metadata.txt")], 1, 11.0
    )
    assert len(tile_specs) == 1
