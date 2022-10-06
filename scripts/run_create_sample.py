import argparse

from prefect import Flow, Parameter, task

from sbem.experiment.Experiment_v2 import Experiment
from sbem.record_v2.Sample import Sample


@task()
def load_experiment(path: str):
    return Experiment.load(path=path)


@task()
def create_sample(exp: Experiment, name: str, description: str):
    assert " " not in name, "Name contains spaces."
    assert exp.get_sample(name) is None, "Sample exists already."
    Sample(
        experiment=exp,
        name=name,
        description=description,
        documentation="",
        aligned_data="",
    )

    exp.save(overwrite=True)


with Flow("Add Sample to Experiment") as flow:
    exp_path = Parameter("exp_path", default="/path/to/experiment.yaml")
    name = Parameter("name", default="Sample")
    description = Parameter("description", default="Experiment to answer the question.")

    exp = load_experiment(path=exp_path)
    create_sample(exp=exp, name=name, description=description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path")
    parser.add_argument("--name")
    parser.add_argument("--description")
    args = parser.parse_args()

    kwargs = {
        "exp_path": args.exp_path,
        "name": args.name,
        "description": args.description,
    }

    flow.run(parameters=kwargs)
