import argparse

from prefect import Flow, Parameter, task

from sbem.experiment.Experiment_v2 import Experiment
from sbem.experiment.parse_utils import parse_and_add_sections


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


with Flow("Add Sample to Experiment") as flow:
    exp_path = Parameter("exp_path", default="/path/to/experiment.yaml")
    sample_name = Parameter("sample_name", default="Sample")
    sbem_root_dir = Parameter("sbem_root_dir", default="/path/to/sbem/acquisition")
    acquisition = Parameter("acquisition", default="run_0")
    tile_grid = Parameter("tile_grid", default="g0001")
    thickness = Parameter("thickness", default=25.0)
    resolution_xy = Parameter("resolution_xy", default=11.0)
    tile_width = Parameter("tile_width", default=3072)
    tile_height = Parameter("tile_height", default=2304)
    tile_overlap = Parameter("tile_overlap", default=200)

    exp = load_experiment(path=exp_path)
    add_sections(
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

    flow.run(parameters=kwargs)
