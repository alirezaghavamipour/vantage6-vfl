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


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2) -> dict:
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


def vanilla_train_splitvfl_bottom_run(matching_method: str = "exact", fuzzy_threshold: int = 2):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) splitVFL training function.

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then trains its own small
    bottom model (Dense -> ReLU) - sends its per-row embedding vector to
    the label party IN THE CLEAR each epoch, receives a gradient vector
    IN THE CLEAR back and backpropagates through its own ReLU to update
    locally. Same protocol as vanilla_train_splitvflc_bottom_run -
    reuses the identical underlying code, only the dataset/columns
    differ (splitVFL moves some columns from a feature party to the
    label party, same split as aggVFL). Invoked internally by
    'central_train_splitvfl_vanilla', not meant to be run standalone.
    """
    return _run_job("vanilla_train_splitvfl_bottom_run", matching_method, fuzzy_threshold)


def vanilla_train_splitvfl_top_run(matching_method: str = "exact", fuzzy_threshold: int = 2):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) splitVFL training function.

    Label party: unlike splitVFLc, this party now also holds its own
    features (not just the label) - it re-aligns local rows (reusing
    the same Private PSI logic), trains its own bottom model (Dense ->
    ReLU) on those features (no network round-trip needed for this
    part, since it already holds both the features and the top model),
    concatenates its own embedding with both feature parties'
    embeddings received IN THE CLEAR each epoch, and trains its top
    model (Dense -> Sigmoid) on the combined 3-way concatenation -
    computes the prediction and error using its own real labels, and
    sends the resulting gradient vector back to each feature party IN
    THE CLEAR. Invoked internally by 'central_train_splitvfl_vanilla',
    not meant to be run standalone.
    """
    return _run_job("vanilla_train_splitvfl_top_run", matching_method, fuzzy_threshold)
