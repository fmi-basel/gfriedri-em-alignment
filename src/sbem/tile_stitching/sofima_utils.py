import functools as ft
from os.path import exists, join

import jax
import jax.numpy as jnp
import numpy as np
import ray
from sofima import flow_utils, mesh, stitch_elastic, stitch_rigid, warp

from sbem.record.SectionRecord import SectionRecord


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
    section: SectionRecord,
    stride: int,
    overlaps_x: tuple,
    overlaps_y: tuple,
    min_overlap: int,
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
):
    tile_space = section.tile_id_map.shape
    tile_map = section.get_tile_data_map()
    cx, cy = stitch_rigid.compute_coarse_offsets(
        tile_space,
        tile_map,
        overlaps_xy=(overlaps_x, overlaps_y),
        min_overlap=min_overlap,
    )
    coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)
    cx = np.squeeze(cx, axis=1)
    cy = np.squeeze(cy, axis=1)
    fine_x, offsets_x = stitch_elastic.compute_flow_map(
        tile_map, cx, 0, stride=(stride, stride),
        patch_size=patch_size, batch_size=batch_size
    )
    fine_y, offsets_y = stitch_elastic.compute_flow_map(
        tile_map, cy, 1, stride=(stride, stride),
        patch_size=patch_size, batch_size=batch_size
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

    mesh_path = join(section.save_dir, "meshes.npz")
    np.savez(mesh_path, **{str(k): v for k, v in meshes.items()})
    return mesh_path


@ray.remote(num_gpus=1 / 6.0, max_calls=1)
def run_sofima(
    section: SectionRecord,
    stride: int,
    overlaps_x: tuple,
    overlaps_y: tuple,
    min_overlap: int,
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
):
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1 / 6.0)

    from sbem.tile_stitching.sofima_utils import register_tiles

    try:
        register_tiles(
            section,
            stride=stride,
            overlaps_x=overlaps_x,
            overlaps_y=overlaps_y,
            min_overlap=min_overlap,
            patch_size=patch_size,
            batch_size=batch_size,
            min_peak_ratio=min_peak_ratio,
            min_peak_sharpness=min_peak_sharpness,
            max_deviation=max_deviation,
            max_magnitude=max_magnitude,
            min_patch_size=min_patch_size,
            max_gradient=max_gradient,
            reconcile_flow_max_deviation=reconcile_flow_max_deviation,
            integration_config=integration_config,
        )
        return section
    except Exception as e:
        print(f"Encounter error in section {section.save_dir}.")
        print(e)
        return section


def render_tiles(
    section: SectionRecord,
    stride,
    margin=50,
    parallelism=1,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
):
    tile_map = section.get_tile_data_map()
    if exists(join(section.save_dir, "meshes.npz")):
        data = np.load(join(section.save_dir, "meshes.npz"))
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


@ray.remote(num_cpus=1)
def run_warp_and_save(
    section: SectionRecord,
    stride: int,
    margin: int = 50,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
):
    from sbem.tile_stitching.sofima_utils import render_tiles

    stitched, mask = render_tiles(
        section,
        stride=stride,
        margin=margin,
        parallelism=1,
        use_clahe=use_clahe,
        clahe_kwargs=clahe_kwargs,
    )

    if stitched is not None and mask is not None:
        section.write_stitched(stitched=stitched, mask=mask)

    return section
