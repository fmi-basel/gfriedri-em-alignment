import argparse
import json
from os.path import join
from typing import Dict

import git
from prefect_dask import DaskTaskRunner

from prefect import flow, get_run_logger, task
from prefect.blocks.system import String
from sbem.experiment.Experiment_v2 import Experiment
from sbem.experiment.parse_utils import parse_and_add_sections
from sbem.utils.env import save_conda_env
from sbem.utils.system import save_system_information


@task()
def load_experiment(path: str):
    return Experiment.load(path=path)


@task()
def add_sections(
    exp: Experiment,
    sample_name: str,
    sbem_root_dir: str,
    acquisition: str,
    tile_grid: str,
    thickness: float,
    resolution_xy: float,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
):
    assert exp.get_sample(sample_name) is not None, "Sample does not exists."

    sample = exp.get_sample(sample_name)

    parse_and_add_sections(
        sbem_root_dir=sbem_root_dir,
        sample=sample,
        acquisition=acquisition,
        tile_grid=tile_grid,
        thickness=thickness,
        resolution_xy=resolution_xy,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_overlap=tile_overlap,
        overwrite=True,
    )

    exp.save(overwrite=True)


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_add-sections.json")
    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment, name: str):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit(f"Add sections to sample '{name}'.", author=exp._git_author)


async def get_prologue():
    return await String.load("log-slurm-job").value


@flow(
    name="Add Sections",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 2,
            "memory": "12 GB",
            "walltime": "01:00:00",
            "worker_extra_args": ["--lifetime", "55m", "--lifetime-stagger", "5m"],
            "job_script_prologue": get_prologue(),
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 1,
        },
    ),
)
def add_sections_to_sample_flow(
    exp_path: str = "/path/to/experiment.yaml",
    sample_name: str = "Sample",
    sbem_root_dir: str = "/path/to/sbem/acquisition",
    acquisition: str = "run_0",
    tile_grid: str = "g0001",
    thickness: float = 25.0,
    resolution_xy: float = 11.0,
    tile_width: int = 3072,
    tile_height: int = 2304,
    tile_overlap: int = 200,
):
    params = dict(locals())
    exp = load_experiment.submit(path=exp_path).result()
    as_task = add_sections.submit(
        exp=exp,
        sample_name=sample_name,
        sbem_root_dir=sbem_root_dir,
        acquisition=acquisition,
        tile_grid=tile_grid,
        thickness=thickness,
        resolution_xy=resolution_xy,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_overlap=tile_overlap,
    )

    save_env = save_conda_env.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    save_sys = save_system_information.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    commit_changes.submit(
        exp=exp,
        name=sample_name,
        wait_for=[exp, as_task, save_env, save_sys, run_context],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path")
    parser.add_argument("--sample_name")
    parser.add_argument("--sbem_root_dir")
    parser.add_argument("--acquisition")
    parser.add_argument("--tile_grid")
    parser.add_argument("--thickness")
    parser.add_argument("--resolution_xy")
    parser.add_argument("--tile_width")
    parser.add_argument("--tile_height")
    parser.add_argument("--tile_overlap")
    args = parser.parse_args()

    kwargs = {
        "exp_path": args.exp_path,
        "sample_name": args.sample_name,
        "sbem_root_dir": args.sbem_root_dir,
        "acquisition": args.acquisition,
        "tile_grid": args.tile_grid,
        "thickness": float(args.thickness),
        "resolution_xy": float(args.resolution_xy),
        "tile_width": int(args.tile_width),
        "tile_height": int(args.tile_height),
        "tile_overlap": int(args.tile_overlap),
    }

    add_sections_to_sample_flow(
        exp_path=args.exp_path,
        sample_name=args.sample_name,
        sbem_root_dir=args.sbem_root_dir,
        acquisition=args.acquisition,
        tile_grid=args.tile_grid,
        thickness=float(args.thickness),
        resolution_xy=float(args.resolution_xy),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        tile_overlap=int(args.tile_overlap),
    )
