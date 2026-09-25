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
    """NOT PRIVATE - deliberately insecure plaintext baseline that is a
    true unencrypted mirror of the secure splitVFLc circuit's model
    (one joint Dense+ReLU hidden layer over every party's concatenated
    features, then Dense+Sigmoid).

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then holds its own row-slice
    of the ONE shared Dense1 weight matrix (not a separate per-party
    bottom model) - sends the label party a linear partial
    pre-activation IN THE CLEAR each epoch (no local nonlinearity; the
    ReLU is shared and lives on the label party, matching where the
    joint circuit applies it), and receives the SAME shared gradient
    matrix back to update its own weight slice. See
    mpc_daemon_client_v2.py's module comment above
    _vanilla_joint_feature_party_train for why this is mathematically
    equivalent to the joint circuit's matmul (a joint matrix product is
    separable into a sum of per-party partial products when each party
    holds a matching row-slice of the weight matrix). Invoked
    internally by 'central_train_splitvflc_vanilla', not meant to be
    run standalone.
    """
    return _run_job("vanilla_train_splitvflc_bottom_run", matching_method, fuzzy_threshold, run_id)


def vanilla_train_splitvflc_top_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """NOT PRIVATE - deliberately insecure plaintext baseline that is a
    true unencrypted mirror of the secure splitVFLc circuit's model.

    Label party: re-aligns local rows (reusing the same Private PSI
    logic), sums both feature parties' partial pre-activations plus its
    own bias, applies the ONE shared ReLU, then runs its Dense+Sigmoid
    output layer - computes the prediction and error using its own real
    labels, and sends the SAME shared gradient matrix back to both
    feature parties (not a per-party slice, since the nonlinearity and
    the weight matrix it feeds are shared, not partitioned by party).
    Invoked internally by 'central_train_splitvflc_vanilla', not meant
    to be run standalone.
    """
    return _run_job("vanilla_train_splitvflc_top_run", matching_method, fuzzy_threshold, run_id)
