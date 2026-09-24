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
    info(f"Vanilla splitVFLc train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(VANILLA_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Vanilla splitVFLc train: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Vanilla splitVFLc train: host daemon did not respond to job {job_id} within {VANILLA_TIMEOUT}s")


def vanilla_train_splitvflc_bottom_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) splitVFLc training function.

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then runs its own small local
    "bottom model" (one trainable Dense layer + ReLU) - unlike aggVFLc's
    vanilla worker, which sends a single partial linear score, this
    sends a per-row EMBEDDING VECTOR to the label party IN THE CLEAR
    each epoch, and receives a gradient VECTOR (not a single error
    value) back, which it backpropagates through its own ReLU to update
    its local weights. This is the standard (insecure) SplitNN design,
    generally considered a weaker privacy story than aggVFLc's vanilla
    since embeddings can leak more information than a single score.
    Invoked internally by 'central_train_splitvflc_vanilla', not meant
    to be run standalone.
    """
    return _run_job("vanilla_train_splitvflc_bottom_run", matching_method, fuzzy_threshold, run_id)


def vanilla_train_splitvflc_top_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) splitVFLc training function.

    Label party: re-aligns local rows (reusing the same Private PSI
    logic), then runs its own small "top model" (one trainable Dense
    layer + Sigmoid) on the concatenation of both feature parties'
    embeddings, received IN THE CLEAR each epoch - computes the
    prediction and error using its own real labels, backpropagates
    through its top layer, and sends the resulting gradient vector back
    to each feature party IN THE CLEAR so they can backpropagate
    through their own bottom models. Invoked internally by
    'central_train_splitvflc_vanilla', not meant to be run standalone.
    """
    return _run_job("vanilla_train_splitvflc_top_run", matching_method, fuzzy_threshold, run_id)
