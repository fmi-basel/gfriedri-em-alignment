import os
from time import sleep

from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner


@task()
def dummy_task():
    logger = get_run_logger()
    logger.info(f"I am the task and run on {os.environ['SLURM_JOB_ID']}.")
    for i in range(10):
        sleep(10)
        logger.info(f"I am the task and run on {os.environ['SLURM_JOB_ID']}.")


@flow(
    name="slurm test",
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
def test_flow():
    dummy_task.submit()


if __name__ == "__main__":
    test_flow()
