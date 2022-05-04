import argparse
import configparser

from sofima import mesh

from sbem import parallel_tile_registration
from sbem.experiment import Experiment


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
        "min_peak_ratio": float(register_tiles["min_peak_ratio"]),
        "min_peak_sharpness": float(register_tiles["min_peak_sharpness"]),
        "max_deviation": float(register_tiles["max_deviation"]),
        "max_magnitude": float(register_tiles["max_magnitude"]),
        "min_patch_size": int(register_tiles["min_patch_size"]),
        "max_gradient": float(register_tiles["max_gradient"]),
        "reconcile_flow_max_deviation": float(
            register_tiles["reconcile_flow_max_deviation"]
        ),
        "n_workers": int(register_tiles["n_workers"]),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = config_to_dict(config)

    exp = Experiment()
    exp.load(kwargs["sbem_experiment"])

    block = exp.blocks[kwargs["block"]]
    start_section = kwargs["start_section"]
    end_section = kwargs["end_section"] + 1

    sections = [
        block.sections[(i, kwargs["grid_index"])]
        for i in range(start_section, end_section)
    ]

    integration_config = mesh.IntegrationConfig(
        dt=kwargs["dt"],
        gamma=kwargs["gamma"],
        k0=kwargs["k0"],
        k=kwargs["k"],
        stride=kwargs["stride"],
        num_iters=kwargs["num_iters"],
        max_iters=kwargs["max_iters"],
        stop_v_max=kwargs["stop_v_max"],
        dt_max=kwargs["dt_max"],
        prefer_orig_order=kwargs["prefer_orig_order"],
        start_cap=kwargs["start_cap"],
        final_cap=kwargs["final_cap"],
        remove_drift=kwargs["remove_drift"],
    )

    parallel_tile_registration(
        sections=sections,
        stride=kwargs["stride"],
        batch_size=kwargs["batch_size"],
        min_peak_ratio=kwargs["min_peak_ratio"],
        min_peak_sharpness=kwargs["min_peak_sharpness"],
        max_deviation=kwargs["max_deviation"],
        max_magnitude=kwargs["max_magnitude"],
        min_patch_size=kwargs["min_patch_size"],
        max_gradient=kwargs["max_gradient"],
        reconcile_flow_max_deviation=kwargs["reconcile_flow_max_deviation"],
        integration_config=integration_config,
        n_workers=kwargs["n_workers"],
    )


if __name__ == "__main__":
    main()
