from typing import Dict

import prefect
from prefect.tasks import task


@task()
def get_prefect_context_task() -> Dict:
    """
    Collect prefect context.
    :return:
    """
    return {
        "date": prefect.context.get("date").strftime("%Y-%m-%d %H:%M:%S"),
        "flow_id": prefect.context.get("flow_id"),
        "flow_run_id": prefect.context.get("flow_run_id"),
        "flow_run_version": prefect.context.get("flow_run_version"),
        "flow_run_name": prefect.context.get("flow_run_name"),
    }
