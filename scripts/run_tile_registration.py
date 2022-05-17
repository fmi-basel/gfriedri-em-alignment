import argparse
import configparser
from time import time

import ray
from sofima import mesh

from sbem.experiment import Experiment
from sbem.tile_stitching.sofima_utils import run_sofima, run_warp_and_save


def config_to_dict(config):
    default = config["DEFAULT"]
    register_tiles = config["REGISTER_TILES"]
    mesh_conf = config["MESH_INTEGRATION_CONFIG"]
    warp_conf = config["WARP_CONFIG"]
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
        "use_clahe": warp_conf["use_clahe"] == "True",
        "kernel_size": int(warp_conf["kernel_size"]),
        "clip_limit": float(warp_conf["clip_limit"]),
        "nbins": int(warp_conf["nbins"]),
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

    ray.init(num_gpus=1, num_cpus=20)

    references = []
    for sec in sections:
        reg_obj = run_sofima.remote(
            sec,
            stride=kwargs["stride"],
            overlaps_x=kwargs["overlaps_x"],
            overlaps_y=kwargs["overlaps_y"],
            min_overlap=kwargs["min_overlap"],
            batch_size=kwargs["batch_size"],
            min_peak_ratio=kwargs["min_peak_ratio"],
            min_peak_sharpness=kwargs["min_peak_sharpness"],
            max_deviation=kwargs["max_deviation"],
            max_magnitude=kwargs["max_magnitude"],
            min_patch_size=kwargs["min_patch_size"],
            max_gradient=kwargs["max_gradient"],
            reconcile_flow_max_deviation=kwargs["reconcile_flow_max_deviation"],
            integration_config=integration_config,
        )
        warp_obj = run_warp_and_save.remote(
            reg_obj,
            stride=kwargs["strid"],
            parallelism=1,
            use_clahe=kwargs["use_clahe"],
            clahe_kwargs={
                "kernel_size": kwargs["kernel_size"],
                "clip_limit": kwargs["clip_limit"],
                "nbins": kwargs["nbins"],
            },
        )
        references.append(warp_obj)

    def print_runtime(sec, start_time, decimals=1):
        print(f"Runtime: {time() - start_time:.{decimals}f} seconds, sections:")
        print(*[s.save_dir for s in sec], sep="\n")

    start = time()
    all_sections = []
    while len(references) > 0:
        finished, references = ray.wait(references, num_returns=1)
        sec = ray.get(finished)
        print_runtime(sec, start, 1)
        all_sections.append(sec)


if __name__ == "__main__":
    main()
