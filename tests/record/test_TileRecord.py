from os.path import join

import numpy as np
from numpy.testing import assert_array_equal
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from tifffile import imsave


def test_tile_record(tmpdir):
    img = np.random.rand(127, 193)
    tile_id = 10
    img_path = join(tmpdir, f"tile_{tile_id}.tif")
    imsave(img_path, img)

    section = SectionRecord(None, 1, 1)
    tile = TileRecord(
        section=section, path=img_path, tile_id=tile_id, x=34, y=94, resolution_xy=2.5
    )

    tile_data = tile.get_tile_data()
    assert_array_equal(img, tile_data)

    tile_dict = tile.get_tile_dict()
    assert tile_dict["path"] == img_path
    assert tile_dict["tile_id"] == tile_id
    assert tile_dict["x"] == 34
    assert tile_dict["y"] == 94
    assert tile_dict["resolution_xy"] == 2.5
