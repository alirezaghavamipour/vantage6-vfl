import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# No cryptographic overhead at all (plain sockets, plaintext floats) -
# a full 200-epoch run over 171 rows takes seconds, not the ~130s the
# real MPC version needs. Generous headroom is still kept for the PSI
# alignment step this reuses, which is the same cost as every other
# function here.
VANILLA_TIMEOUT = 1800

SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None) -> dict:
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "action": action,
        "method": matching_method,
        "fuzzy_threshold": fuzzy_threshold,
    }
    # run_id ties every job dispatched by one orchestrator call together
    # across however many hosts they land on - job_id alone is only
    # unique to this one job, with nothing connecting it to its siblings
    # from the same run. Optional/None for direct/manual invocation
    # outside an orchestrator.
    if run_id:
        job["run_id"] = run_id
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Vanilla train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(VANILLA_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Vanilla train: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Vanilla train: host daemon did not respond to job {job_id} within {VANILLA_TIMEOUT}s")


def vanilla_train_worker_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against vantage6-vfl's real (Rep3 MPC) aggVFLc training only.

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then runs standard (non-MPC)
    vertical logistic regression - each epoch it computes its own
    partial linear score and sends it to the label party IN THE CLEAR,
    then receives the prediction error IN THE CLEAR back and updates
    its own weights locally. The label party learns this party's
    partial score every epoch; this party learns
    (prediction - true label) every epoch - a well-documented Direct
    Label Inference risk. Invoked internally by
    'central_train_aggvflc_vanilla', not meant to be run standalone.
    """
    return _run_job("vanilla_train_worker_run", matching_method, fuzzy_threshold, run_id)


def vanilla_train_coordinator_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against vantage6-vfl's real (Rep3 MPC) aggVFLc training only.

    Label party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then coordinates standard
    (non-MPC) vertical logistic regression training directly with each
    feature party - combines their plaintext partial scores, computes
    the sigmoid and prediction error using its own real labels, and
    broadcasts that error back to every feature party IN THE CLEAR each
    epoch. Invoked internally by 'central_train_aggvflc_vanilla', not
    meant to be run standalone.
    """
    return _run_job("vanilla_train_coordinator_run", matching_method, fuzzy_threshold, run_id)
