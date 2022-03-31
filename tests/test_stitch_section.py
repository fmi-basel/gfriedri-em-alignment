from generate_tile_id_map import get_section_range


def test_get_section_range():
    tile_spec_list = [{"z": 3}, {"z": 4}, {"z": 99}]

    min, max = get_section_range(tile_spec_list)

    assert min == 3
    assert max == 99
