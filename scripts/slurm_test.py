import argparse
from os.path import join
from time import sleep
from typing import Dict

from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner

from sbem.experiment.Experiment_v2 import Experiment


@task()
def load_experiment(path: str):
    sleep(2)


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
    sleep(15)


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    sleep(2)

    logger.info("Saved flow parameters to ....")


@task()
def commit_changes(exp: Experiment, name: str):
    sleep(2)
    logger = get_run_logger()
    logger.info("commit stuff")


@flow(
    name="Slurm Test",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 2,
            "memory": "12 GB",
            "walltime": "01:00:00",
            "worker_extra_args": ["--lifetime", "55m", "--lifetime-stagger", "5m"],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 1,
        },
    ),
)
def test_flow(
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

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    commit_changes.submit(
        exp=exp,
        name=sample_name,
        wait_for=[exp, as_task, run_context],
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

    test_flow(
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
