import argparse
import json
from os import mkdir
from os.path import exists, join
from typing import Dict

import git
from prefect.flows import flow
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.tasks import task

from sbem.experiment.Experiment_v2 import Experiment
from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation
from sbem.utils.env import save_conda_env
from sbem.utils.system import save_system_information


@task()
def create_experiment(name: str, description: str, root_dir: str) -> Experiment:
    assert " " not in name, "Name contains spaces."
    exp = Experiment(
        name=name,
        description=description,
        documentation="",
        authors=[Author(name="", affiliation="")],
        root_dir=root_dir,
        exist_ok=True,
        cite=[Citation(doi="", text="", url="")],
    )
    exp.save()

    if not exists(join(exp.get_root_dir(), exp.get_name(), "processing")):
        mkdir(join(exp.get_root_dir(), exp.get_name(), "processing"))

    return exp


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_create-experiment.json")
    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit("Create experiment.", author=exp._git_author)


@flow(name="Create Experiment", task_runner=SequentialTaskRunner())
def create_experiment_flow(
    name: str = "experiment",
    description: str = "Experiment to answer " "questions.",
    root_dir: str = "/tungstenfs/scratch/gmicro_sem",
):
    params = dict(locals())
    exp = create_experiment(name=name, description=description, root_dir=root_dir)

    save_env = save_conda_env(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    save_sys = save_system_information(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    commit_changes(exp=exp, wait_for=[exp, save_env, save_sys, run_context])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--description")
    parser.add_argument("--root_dir")
    args = parser.parse_args()

    kwargs = {
        "name": args.name,
        "description": args.description,
        "root_dir": args.root_dir,
    }

    create_experiment_flow(
        name=args.name, description=args.description, root_dir=args.root_dir
    )
