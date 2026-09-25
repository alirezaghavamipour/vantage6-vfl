import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

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
    info(f"Vanilla splitVFL train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(VANILLA_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Vanilla splitVFL train: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Vanilla splitVFL train: host daemon did not respond to job {job_id} within {VANILLA_TIMEOUT}s")


def vanilla_train_splitvfl_bottom_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure plaintext baseline that is a
    true unencrypted mirror of the secure splitVFL circuit's model (one
    joint Dense+ReLU hidden layer over every party's concatenated
    features, then Dense+Sigmoid).

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then holds its own row-slice
    of the ONE shared Dense1 weight matrix - sends a linear partial
    pre-activation IN THE CLEAR each epoch, receives the SAME shared
    gradient matrix back to update its own slice. Same protocol as
    vanilla_train_splitvflc_bottom_run - reuses the identical
    underlying code, only the dataset/columns differ (splitVFL moves
    some columns from a feature party to the label party, same split as
    aggVFL). Invoked internally by 'central_train_splitvfl_vanilla',
    not meant to be run standalone.
    """
    return _run_job("vanilla_train_splitvfl_bottom_run", matching_method, fuzzy_threshold, run_id)


def vanilla_train_splitvfl_top_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure plaintext baseline that is a
    true unencrypted mirror of the secure splitVFL circuit's model.

    Label party: unlike splitVFLc, this party now also holds its own
    features (not just the label) - it re-aligns local rows (reusing
    the same Private PSI logic), holds its own row-slice of the shared
    Dense1 weight matrix (computed and updated purely locally, no
    network round-trip needed since it already holds both this slice
    and the top model), sums its own partial pre-activation with both
    feature parties' partials received IN THE CLEAR each epoch, applies
    the ONE shared ReLU, then runs its Dense+Sigmoid output layer -
    computes the prediction and error using its own real labels, and
    sends the SAME shared gradient matrix back to each feature party IN
    THE CLEAR. Invoked internally by 'central_train_splitvfl_vanilla',
    not meant to be run standalone.
    """
    return _run_job("vanilla_train_splitvfl_top_run", matching_method, fuzzy_threshold, run_id)
