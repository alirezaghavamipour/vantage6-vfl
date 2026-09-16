import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")


def _run_job(action: str, timeout: int = 120) -> dict:
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": action}
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"PSI: submitted job {job_id} ({action}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(timeout):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"PSI: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"PSI: host daemon did not respond to job {job_id} within {timeout}s")


def psi_client_share():
    """Feature/label party: share local entity presence vector into the MPC computation."""
    return _run_job("psi_client_run")


def psi_party_run():
    """Computing party: run this party's role in the Rep3 PSI computation."""
    return _run_job("psi_party_run")
