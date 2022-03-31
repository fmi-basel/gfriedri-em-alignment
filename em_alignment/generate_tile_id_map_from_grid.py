import os

import numpy as np


def tile_id_map_from_grid(grid_size):
    # nrow, ncol = grid_size
    tile_id_list = np.arange(grid_size[0] * grid_size[1])
    tile_id_map = np.reshape(tile_id_list, grid_size)
    return tile_id_map


def crop_tile_id_map(full_tile_id_map, active_tiles):
    active_tiles = sorted(active_tiles)
    tile_pos = [np.where(full_tile_id_map == t) for t in active_tiles]
    tile_pos = np.array(tile_pos).squeeze()
    x1 = tile_pos[:, 0].min()
    x2 = tile_pos[:, 0].max()
    y1 = tile_pos[:, 1].min()
    y2 = tile_pos[:, 1].max()
    tile_id_map = full_tile_id_map[x1 : x2 + 1, y1 : y2 + 1]
    return tile_id_map


if __name__ == "__main__":
    stack_name = "example_stack"
    outdir = os.path.join("some_directory", stack_name)
    file_name = "tile_id_map_from_grid.npy"
    grid_size = (42, 32)
    active_tiles = [
        364,
        365,
        366,
        367,
        394,
        395,
        396,
        397,
        398,
        399,
        400,
        401,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        458,
        459,
        460,
        461,
        462,
        463,
        464,
        465,
        466,
        490,
        491,
        492,
        493,
        494,
        495,
        496,
        497,
        498,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        554,
        555,
        556,
        557,
        558,
        559,
        560,
        561,
        562,
        586,
        587,
        588,
        589,
        590,
        591,
        592,
        593,
        594,
        619,
        620,
        621,
        622,
        623,
        624,
        625,
        626,
        652,
        653,
        654,
        655,
        656,
        657,
        658,
    ]
    full_tile_id_map = tile_id_map_from_grid(grid_size)
    tile_id_map = crop_tile_id_map(full_tile_id_map, active_tiles)
    np.save(os.path.join(outdir, file_name), tile_id_map)
