import argparse

from prefect import Flow, Parameter, task

from sbem.experiment.Experiment_v2 import Experiment
from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation


@task()
def create_experiment(name: str, description: str, root_dir: str):
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


with Flow("Create Experiment") as flow:
    name = Parameter("name", default="experiment")
    description = Parameter("description", default="Experiment to answer the question.")
    root_dir = Parameter("root_dir", default="/tungstenfs/scratch/gmicro_sem")

    create_experiment(name=name, description=description, root_dir=root_dir)

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

    flow.run(parameters=kwargs)
