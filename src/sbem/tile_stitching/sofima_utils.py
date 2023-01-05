import functools as ft
import logging
from os.path import exists, join

import jax
import jax.numpy as jnp
import numpy as np
from sofima import flow_utils, mesh, stitch_elastic, stitch_rigid, warp

from sbem.experiment.Experiment import Experiment
from sbem.record.Section import Section


def default_mesh_integration_config(stride: int = 20, k0: float = 0.01, k: float = 0.1):
    return mesh.IntegrationConfig(
        dt=0.001,
        gamma=0.0,
        k0=k0,
        k=k,
        stride=stride,
        num_iters=1000,
        max_iters=20000,
        stop_v_max=0.001,
        dt_max=100,
        prefer_orig_order=True,
        start_cap=0.1,
        final_cap=10.0,
        remove_drift=True,
    )


def default_sofima_config():
    return {
        "batch_size": 4,
        "min_peak_ratio": 1.4,
        "min_peak_sharpness": 1.4,
        "max_deviation": 5,
        "max_magnitude": 0,
        "min_patch_size": 10,
        "max_gradient": -1,
        "reconcile_flow_max_deviation": -1,
        "integration_config": default_mesh_integration_config(),
    }


def register_tiles(
    section: Section,
    section_dir: str,
    stride: int,
    overlaps_x: tuple,
    overlaps_y: tuple,
    min_overlap: int,
    min_range: tuple = (10, 100, 0),
    patch_size: tuple = (120, 120),
    batch_size: int = 8000,
    min_peak_ratio: float = 1.4,
    min_peak_sharpness: float = 1.4,
    max_deviation: int = 5,
    max_magnitude: int = 0,
    min_patch_size: int = 10,
    max_gradient: float = -1,
    reconcile_flow_max_deviation: float = -1,
    integration_config: mesh.IntegrationConfig = default_mesh_integration_config(),
    logger=logging.getLogger("load_sections"),
):
    tim_path = join(
        section_dir,
        "tile_id_map.json",
    )
    tile_space = section.get_tile_id_map(path=tim_path).shape
    tile_map = section.get_tile_data_map(path=tim_path, indexing="xy")
    cx, cy = stitch_rigid.compute_coarse_offsets(
        tile_space,
        tile_map,
        overlaps_xy=(overlaps_x, overlaps_y),
        min_overlap=min_overlap,
        min_range=min_range,
    )
    coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)
    cx = np.squeeze(cx, axis=1)
    cy = np.squeeze(cy, axis=1)

    if np.isinf(cx).any() or np.isinf(cy).any():
        msg = (
            f"register_tiles: Inf in coarse mesh. Coarse rigid registration "
            f"failed. Section number {section.get_section_num()}."
        )
        long_msg = f"{msg}\ncx: {np.array2string(cx)}\ncy: {np.array2string(cy)}"
        logger.error(long_msg)
        raise ValueError(msg)

    fine_x, offsets_x = stitch_elastic.compute_flow_map(
        tile_map,
        cx,
        0,
        stride=(stride, stride),
        patch_size=patch_size,
        batch_size=batch_size,
    )
    fine_y, offsets_y = stitch_elastic.compute_flow_map(
        tile_map,
        cy,
        1,
        stride=(stride, stride),
        patch_size=patch_size,
        batch_size=batch_size,
    )

    fine_x = {
        k: flow_utils.clean_flow(
            v[:, np.newaxis, ...],
            min_peak_ratio,
            min_peak_sharpness,
            max_deviation,
            max_magnitude,
        )[:, 0, :, :]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.clean_flow(
            v[:, np.newaxis, ...],
            min_peak_ratio,
            min_peak_sharpness,
            max_deviation,
            max_magnitude,
        )[:, 0, :, :]
        for k, v in fine_y.items()
    }

    fine_x = {
        k: flow_utils.reconcile_flows(
            [v[:, np.newaxis, ...]],
            min_patch_size,
            max_gradient,
            reconcile_flow_max_deviation,
        )[:, 0, :, :]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.reconcile_flows(
            [v[:, np.newaxis, ...]],
            min_patch_size,
            max_gradient,
            reconcile_flow_max_deviation,
        )[:, 0, :, :]
        for k, v in fine_y.items()
    }

    data_x = (cx, fine_x, offsets_x)
    data_y = (cy, fine_y, offsets_y)

    fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
        data_x, data_y, tile_map, coarse_mesh[:, 0, ...], stride=(stride, stride)
    )

    @jax.jit
    def prev_fn(x):
        target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx, fy=fy)
        x = jax.vmap(target_fn)(nbors)
        return jnp.transpose(x, [1, 0, 2, 3])

    x, ekin, t = mesh.relax_mesh(x, None, integration_config, prev_fn=prev_fn)
    # Unpack meshes into a dictionary.
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    meshes = {idx_to_key[i]: np.array(x[:, i : i + 1]) for i in range(x.shape[1])}

    mesh_path = join(
        section_dir,
        "meshes.npz",
    )
    np.savez(mesh_path, **{str(k): v for k, v in meshes.items()})
    section.set_alignment_mesh(mesh_path)
    return mesh_path


def render_tiles(
    section: Section,
    section_dir: str,
    stride,
    margin=50,
    parallelism=1,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
):
    path = join(
        section_dir,
        "tile_id_map.json",
    )
    tile_map = section.get_tile_data_map(path=path, indexing="xy")
    mesh_path = join(
        section_dir,
        "meshes.npz",
    )
    if exists(mesh_path):
        data = np.load(mesh_path)
        meshes = {tuple(int(i) for i in k[1:-1].split(",")): v for k, v in data.items()}
        # Warp the tiles into a single image.
        stitched, mask = warp.render_tiles(
            tile_map,
            meshes,
            stride=(stride, stride),
            margin=margin,
            parallelism=parallelism,
            use_clahe=use_clahe,
            clahe_kwargs=clahe_kwargs,
        )

        return stitched, mask
    else:
        return None, None


def load_sections(exp: Experiment, sample_name: str, tile_grid_num: int):
    sample = exp.get_sample(name=sample_name)

    start_section = sample.get_min_section_num(tile_grid_num=tile_grid_num)
    end_section = sample.get_max_section_num(tile_grid_num=tile_grid_num)
    return sample.get_section_range(
        start_section_num=start_section,
        end_section_num=end_section,
        tile_grid_num=tile_grid_num,
        include_skipped=False,
    )
