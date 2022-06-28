import argparse
import configparser

from prefect import Flow, Parameter, unmapped
from prefect.executors import LocalDaskExecutor

from sbem.tile_stitching.sofima_utils import (
    build_integration_config,
    load_sections,
    run_sofima,
)


def config_to_dict(config):
    default = config["DEFAULT"]
    register_tiles = config["REGISTER_TILES"]
    mesh_conf = config["MESH_INTEGRATION_CONFIG"]
    kwargs = {
        "sbem_experiment": default["sbem_experiment"],
        "block": default["block"],
        "grid_index": int(default["grid_index"]),
        "start_section": int(register_tiles["start_section"]),
        "end_section": int(register_tiles["end_section"]),
        "batch_size": int(register_tiles["batch_size"]),
        "stride": int(register_tiles["stride"]),
        "overlaps_x": tuple(int(o) for o in register_tiles["overlaps_x"].split(",")),
        "overlaps_y": tuple(int(o) for o in register_tiles["overlaps_y"].split(",")),
        "min_overlap": int(register_tiles["min_overlap"]),
        "patch_size": tuple(int(p) for p in register_tiles["patch_size"].split(",")),
        "min_peak_ratio": float(register_tiles["min_peak_ratio"]),
        "min_peak_sharpness": float(register_tiles["min_peak_sharpness"]),
        "max_deviation": float(register_tiles["max_deviation"]),
        "max_magnitude": float(register_tiles["max_magnitude"]),
        "min_patch_size": int(register_tiles["min_patch_size"]),
        "max_gradient": float(register_tiles["max_gradient"]),
        "reconcile_flow_max_deviation": float(
            register_tiles["reconcile_flow_max_deviation"]
        ),
        "dt": float(mesh_conf["dt"]),
        "gamma": float(mesh_conf["gamma"]),
        "k0": float(mesh_conf["k0"]),
        "k": float(mesh_conf["k"]),
        "num_iters": int(mesh_conf["num_iters"]),
        "max_iters": int(mesh_conf["max_iters"]),
        "stop_v_max": float(mesh_conf["stop_v_max"]),
        "dt_max": float(mesh_conf["dt_max"]),
        "prefer_orig_order": mesh_conf["prefer_orig_order"] == "True",
        "start_cap": float(mesh_conf["start_cap"]),
        "final_cap": float(mesh_conf["final_cap"]),
        "remove_drift": mesh_conf["remove_drift"] == "True",
    }

    return kwargs


with Flow("Tile-Stitching") as flow:
    sbem_experiment = Parameter("sbem_experiment", default="SBEM")
    block = Parameter("block", default="Block")
    start_section = Parameter("start_section", default=0)
    end_section = Parameter("end_section", default=1)
    grid_index = Parameter("grid_index", default=1)

    dt = Parameter("dt", default=0.001)
    gamma = Parameter("gamma", default=0.0)
    k0 = Parameter("k0", default=0.01)
    k = Parameter("k", default=0.1)
    stride = Parameter("stride", default=20)
    num_iters = Parameter("num_iters", default=1000)
    max_iters = Parameter("max_iters", default=20000)
    stop_v_max = Parameter("stop_v_max", default=0.001)
    dt_max = Parameter("dt_max", default=100.0)
    prefer_orig_order = Parameter("prefer_orig_order", default=True)
    start_cap = Parameter("start_cap", default=0.1)
    final_cap = Parameter("final_cap", default=10.0)
    remove_drift = Parameter("remove_drift", default=True)

    overlaps_x = Parameter("overlaps_x", default=[200, 300])
    overlaps_y = Parameter("overlaps_y", default=[200, 300])
    min_overlap = Parameter("min_overlap", default=20)
    patch_size = Parameter("patch_size", default=[120, 120])
    batch_size = Parameter("batch_size", default=8000)
    min_peak_ratio = Parameter("min_peak_ratio", default=1.4)
    min_peak_sharpness = Parameter("min_peak_sharpness", default=1.4)
    max_deviation = Parameter("max_deviation", default=5.0)
    max_magnitude = Parameter("max_magnitude", default=0.0)
    min_patch_size = Parameter("min_patch_size", default=10)
    max_gradient = Parameter("max_gradient", default=-1.0)
    reconcile_flow_max_deviation = Parameter(
        "reconcile_flow_max_deviation", default=-1.0
    )

    n_workers = Parameter("n_workers", default=6)

    sections = load_sections(
        sbem_experiment=sbem_experiment,
        block=block,
        grid_index=grid_index,
        start_section=start_section,
        end_section=end_section,
    )

    integration_config = build_integration_config(
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

    reg_obj = run_sofima.map(
        sections,
        stride=unmapped(stride),
        overlaps_x=unmapped(overlaps_x),
        overlaps_y=unmapped(overlaps_y),
        min_overlap=unmapped(min_overlap),
        patch_size=unmapped(patch_size),
        batch_size=unmapped(batch_size),
        min_peak_ratio=unmapped(min_peak_ratio),
        min_peak_sharpness=unmapped(min_peak_sharpness),
        max_deviation=unmapped(max_deviation),
        max_magnitude=unmapped(max_magnitude),
        min_patch_size=unmapped(min_patch_size),
        max_gradient=unmapped(max_gradient),
        reconcile_flow_max_deviation=unmapped(reconcile_flow_max_deviation),
        integration_config=unmapped(integration_config),
        n_workers=unmapped(n_workers),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = config_to_dict(config)
    kwargs["n_workers"] = 6

    flow.executor = LocalDaskExecutor(num_workers=6)

    flow.run(parameters=kwargs)
