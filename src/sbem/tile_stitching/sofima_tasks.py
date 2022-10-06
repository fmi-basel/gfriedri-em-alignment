import traceback
from os.path import join

import prefect
from prefect import task
from sofima import mesh

from sbem.experiment.Experiment_v2 import Experiment
from sbem.record.SectionRecord import SectionRecord
from sbem.record_v2.Section import Section
from sbem.tile_stitching.sofima_utils import (
    default_mesh_integration_config,
    load_sections,
)


@task()
def run_sofima(
    section: Section,
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
    n_workers=6,
):
    logger = prefect.context.get("logger")
    sec_long_name = join(
        section.get_sample().get_experiment().get_name(),
        section.get_sample().get_name(),
        section.get_name(),
    )
    logger.info(f"Compute mesh for section {sec_long_name}.")

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1 / float(n_workers))

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
        print(f"Encounter error in section {sec_long_name}.")
        print(e)
        tb = traceback.format_exc()
        print(tb)
        return section


@task()
def run_warp_and_save(
    section: SectionRecord,
    stride: int,
    margin: int = 50,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
    parallelism: int = 1,
):
    logger = prefect.context.get("logger")
    logger.info(f"Warp and save section {section.save_dir}.")

    from sbem.tile_stitching.sofima_utils import render_tiles

    stitched, mask = render_tiles(
        section,
        stride=stride,
        margin=margin,
        parallelism=parallelism,
        use_clahe=use_clahe,
        clahe_kwargs=clahe_kwargs,
    )

    if stitched is not None and mask is not None:
        section.write_stitched(stitched=stitched, mask=mask)

    return section


@task()
def load_experiment_task(exp_path: str):
    return Experiment.load(path=exp_path)


@task()
def load_sections_task(exp: Experiment, sample_name: str, tile_grid_num: int):
    return load_sections(exp=exp, sample_name=sample_name, tile_grid_num=tile_grid_num)


@task()
def build_integration_config(
    dt,
    gamma,
    k0,
    k,
    stride,
    num_iters,
    max_iters,
    stop_v_max,
    dt_max,
    prefer_orig_order,
    start_cap,
    final_cap,
    remove_drift,
):
    return mesh.IntegrationConfig(
        dt=dt,
        gamma=gamma,
        k0=k0,
        k=k,
        stride=stride,
        num_iters=num_iters,
        max_iters=max_iters,
        stop_v_max=stop_v_max,
        dt_max=dt_max,
        prefer_orig_order=prefer_orig_order,
        start_cap=start_cap,
        final_cap=final_cap,
        remove_drift=remove_drift,
    )
