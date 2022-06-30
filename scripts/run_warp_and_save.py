import argparse
import configparser

from prefect import Flow, Parameter, unmapped
from prefect.executors import LocalDaskExecutor

from sbem.tile_stitching.sofima_tasks import load_sections, run_warp_and_save


def config_to_dict(config):
    default = config["DEFAULT"]
    warp_conf = config["WARP_CONFIG"]
    kwargs = {
        "sbem_experiment": default["sbem_experiment"],
        "block": default["block"],
        "grid_index": int(default["grid_index"]),
        "start_section": int(config["REGISTER_TILES"]["start_section"]),
        "end_section": int(config["REGISTER_TILES"]["end_section"]),
        "stride": int(config["REGISTER_TILES"]["stride"]),
        "margin": int(warp_conf["margin"]),
        "use_clahe": warp_conf["use_clahe"] == "True",
        "kernel_size": int(warp_conf["kernel_size"]),
        "clip_limit": float(warp_conf["clip_limit"]),
        "nbins": int(warp_conf["nbins"]),
        "parallelism": int(warp_conf["parallelism"]),
        "cpus": int(config["SLURM"]["cpus_warp"]),
    }

    return kwargs


with Flow("Section-Warping-and-Saving") as flow:
    sbem_experiment = Parameter("sbem_experiment", default="SBEM")
    block = Parameter("block", default="Block")
    start_section = Parameter("start_section", default=0)
    end_section = Parameter("end_section", default=1)
    grid_index = Parameter("grid_index", default=1)

    stride = Parameter("stride", default=20)

    margin = Parameter("margin", default=20)
    use_clahe = Parameter("use_clahe", default=True)
    kernel_Size = Parameter("kernel_size", default=1024)
    clip_limit = Parameter("clip_limit", default=0.01)
    nbins = Parameter("nbins", default=256)
    parallelism = Parameter("parallelism", default=4)

    sections = load_sections(
        sbem_experiment=sbem_experiment,
        block=block,
        grid_index=grid_index,
        start_section=start_section,
        end_section=end_section,
    )

    warp_obj = run_warp_and_save.map(
        sections,
        stride=unmapped(stride),
        margin=unmapped(margin),
        use_clahe=unmapped(use_clahe),
        clahe_kwargs=unmapped(
            {
                "kernel_size": kernel_Size,
                "clip_limit": clip_limit,
                "nbins": nbins,
            }
        ),
        parallelism=unmapped(parallelism),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = config_to_dict(config)

    flow.executor = LocalDaskExecutor(
        num_workers=kwargs["cpus"] // kwargs["parallelism"]
    )

    kwargs.pop("cpus")

    flow.run(parameters=kwargs)
