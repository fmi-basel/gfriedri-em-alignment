from os.path import join

import numpy as np
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from tifffile import imsave


def test_section_record(tmpdir):
    block = BlockRecord(None, None, "bloc1", None)

    section_num = 745
    tile_grid_num = 1
    section = SectionRecord(
        block=block,
        section_num=section_num,
        tile_grid_num=tile_grid_num,
        save_dir=tmpdir,
    )

    data = np.random.rand(93, 84)
    data_path = join(tmpdir, "data.tif")
    imsave(data_path, data)

    # tile-map
    #  3, 18,
    # 20,  4
    tile_03 = TileRecord(
        section, path=data_path, tile_id=3, x=0, y=0, resolution_xy=1.2
    )
    section.register_tile(tile_03)
    tile_18 = TileRecord(
        section, path=data_path, tile_id=18, x=10, y=0, resolution_xy=1.2
    )
    section.register_tile(tile_18)
    tile_20 = TileRecord(
        section, path=data_path, tile_id=20, x=0, y=9, resolution_xy=1.2
    )
    section.register_tile(tile_20)
    tile_04 = TileRecord(
        section, path=data_path, tile_id=4, x=10, y=9, resolution_xy=1.2
    )
    section.register_tile(tile_04)

    assert section.get_tile(3) == tile_03
    assert section.get_tile(18) == tile_18
    assert section.get_tile(20) == tile_20
    assert section.get_tile(4) == tile_04

    section.compute_tile_id_map()
    assert section.tile_id_map[0, 0] == 3
    assert section.tile_id_map[0, 1] == 18
    assert section.tile_id_map[1, 0] == 20
    assert section.tile_id_map[1, 1] == 4

    assert section.get_name() == f"s{section_num}_g{tile_grid_num}"

    section.save()
    load_section = SectionRecord(
        block=block, section_num=section_num, tile_grid_num=tile_grid_num, save_dir=None
    )
    load_section.load(join(tmpdir, load_section.get_name()))
    assert section.section_num == load_section.section_num
    assert len(load_section.tile_map) == 4
    assert section.get_tile(3).tile_id == 3
    assert section.get_tile(18).tile_id == 18
    assert section.get_tile(20).tile_id == 20
    assert section.get_tile(3).tile_id == 3

    assert section.get_tile(3).x == 0
    assert section.get_tile(3).y == 0
    assert section.get_tile(3).resolution_xy == 1.2

    assert section.tile_id_map[0, 0] == 3
    assert section.tile_id_map[0, 1] == 18
    assert section.tile_id_map[1, 0] == 20
    assert section.tile_id_map[1, 1] == 4
