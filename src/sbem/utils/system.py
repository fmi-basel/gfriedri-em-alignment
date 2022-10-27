import json
from os.path import join

from distro import distro
from prefect.logging import get_run_logger
from prefect.tasks import task


@task()
def save_system_information(output_dir: str):
    """
    Dump system information into system-info.json.
    :param output_dir:
    :param logger:
    :return:
    """
    logger = get_run_logger()
    outpath = join(output_dir, "system-info.json")
    info = distro.info(pretty=True, best=True)
    with open(outpath, "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"Saved system information to {outpath}.")
