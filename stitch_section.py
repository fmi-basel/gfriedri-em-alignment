# This is a script to run 2D stitching of a single section with SOFIMA
# Adapted from Tim-Oliver Buchholz's code
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import PIL
from skimage.io import imsave

from sofima import stitch_rigid
from sofima import stitch_elastic
from sofima import flow_utils
from sofima import mesh
from sofima import warp

import argparse
import configparser


def _get_basedir(path):
    return os.path.basename(os.path.normpath(path))


def get_tile_file(tile_id, sbem_dir, grid_num, section_num):
    stackname = _get_basedir(sbem_dir)
    grid_str = 'g{:04d}'.format(int(grid_num))
    tile_str = 't{:04d}'.format(tile_id)
    section_str = 's{:05d}'.format(int(section_num))
    filename = '{}_{}_{}_{}.tif'.format(stackname, grid_str, tile_str, section_str)
    tile_file = os.path.join(sbem_dir, 'tiles', grid_str, tile_str, filename)
    return tile_file


def load_image_to_tile_map(tile_id_map, sbem_dir, grid_num, section_num):
    tile_map = {}
    for y in range(tile_id_map.shape[0]):
        for x in range(tile_id_map.shape[1]):
            tile_id = tile_id_map[y, x]
            tile_file = get_tile_file(tile_id, sbem_dir, grid_num, section_num)
            if not os.path.isfile(tile_file):
                continue
            print('Loading {}'.format(tile_file))
            with open(tile_file, 'rb') as fp:
                img = PIL.Image.open(fp)
                tile_map[(x, y)] = np.array(img)
    return tile_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    sofima_config = configparser.ConfigParser()
    sofima_config.read(args.config)


    tile_id_map_file = sofima_config['DEFAULT']['tile_id_map_file']
    tile_id_map = np.load(tile_id_map_file)

    # Load tile images.
    tile_map = load_image_to_tile_map(tile_id_map,
                                      sofima_config['DEFAULT']['sbem_dir'],
                                      sofima_config['DEFAULT']['grid_num'],
                                      sofima_config['DEFAULT']['section_num'])

    tile_space = tile_id_map.shape
    cx, cy = stitch_rigid.compute_coarse_offsets(tile_space, tile_map)

    coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)


    stride = int(sofima_config['DEFAULT']['stride'])
    cx = np.squeeze(cx)
    cy = np.squeeze(cy)
    fine_x, offsets_x = stitch_elastic.compute_flow_map(tile_map, cx, 0, stride=(stride, stride), batch_size=4)
    fine_y, offsets_y = stitch_elastic.compute_flow_map(tile_map, cy, 1, stride=(stride, stride), batch_size=4)

    kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, "max_deviation": 5, "max_magnitude": 0}
    fine_x = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
    fine_y = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}

    kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
    fine_x = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
    fine_y = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}

    data_x = (cx, fine_x, offsets_x)
    data_y = (cy, fine_y, offsets_y)

    fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
        data_x, data_y, tile_map,
        coarse_mesh[:, 0, ...], stride=(stride, stride))

    @jax.jit
    def prev_fn(x):
        target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx, fy=fy)
        x = jax.vmap(target_fn)(nbors)
        return jnp.transpose(x, [1, 0, 2, 3])

    # These default settings are expect to work well in most configurations. Perhaps
    # the most salient parameter is the elasticity ratio k0 / k. The larger it gets,
    # the more the tiles will be allowed to deform to match their neighbors (in which
    # case you might want use aggressive flow filtering to ensure that there are no
    # inaccurate flow vectors). Lower ratios will reduce deformation, which, depending
    # on the initial state of the tiles, might result in visible seams.
    config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=float(sofima_config['DEFAULT']['k0']),
                                    k=float(sofima_config['DEFAULT']['k']), stride=stride,
                                    num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                    dt_max=100, prefer_orig_order=True,
                                    start_cap=0.1, final_cap=10., remove_drift=True)

    x, ekin, t = mesh.relax_mesh(x, None, config, prev_fn=prev_fn)

    # Unpack meshes into a dictionary.
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}

    # Warp the tiles into a single image.
    stitched, mask = warp.render_tiles(tile_map, meshes, stride=(stride, stride))

    section_num = int(sofima_config['DEFAULT']['section_num'])
    imsave(sofima_config['DEFAULT']['outpath'] + '_stitched_section{:05d}.tif'.format(section_num), stitched, compress=6, check_contrast=False)
    imsave(sofima_config['DEFAULT']['outpath'] + '_mask.tif', mask.astype(np.int8), compress=6, check_contrast=False)


if __name__ == "__main__":
    main()
