import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# Measured ~130s for the full 171-row / 13-feature / 200-epoch aggVFLc
# circuit when it succeeds. The host daemon itself retries up to 5 times
# on transient failures (240s each) on both the PSI-alignment step and
# the training step, so the worst case on the daemon side is well over
# 1000s - give the vantage6 task layer enough headroom to never time out
# before the daemon's own retry budget is exhausted.
TRAIN_TIMEOUT = 2000


def _run_job(action: str) -> dict:
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": action}
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Train: submitted job {job_id} ({action}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(TRAIN_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Train: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Train: host daemon did not respond to job {job_id} within {TRAIN_TIMEOUT}s")


def train_client_run():
    """Feature/label party: align rows and share this party's own columns
    into the aggVFLc training computation.

    Internally reuses the same exact-match Private PSI logic as
    vantage6-vfl-psi to independently re-derive the aligned row set
    (no state is shared between separate task submissions), then
    secret-shares this party's own columns for those rows - features
    for a feature party, the label for the label party - so the model
    trains on data that is never combined in plaintext anywhere.
    """
    return _run_job("train_client_run")


def train_party_run():
    """Computing party: run this party's role in the Rep3 aggVFLc
    training computation (fixed-aggregation vertical logistic
    regression; the label party contributes no features of its own).
    This computing party never sees any feature, label, or prediction -
    only its own secret share of the computation.
    """
    return _run_job("train_party_run")
