import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# Same cost profile as aggVFLc (identical circuit shape - 171 rows, 13
# total features, 200 epochs - only which party contributes which
# columns differs), so the same generous headroom applies.
TRAIN_TIMEOUT = 2000

SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None, schema: dict = None) -> dict:
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
    # schema: this run's actual circuit shape, discovered by
    # central_train from the real data - None falls back to the
    # known-working default shape (see ensure_circuit_compiled on the
    # computing party's daemon).
    if schema:
        job["schema"] = schema
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Train (aggVFL): submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(TRAIN_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Train (aggVFL): job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Train (aggVFL): host daemon did not respond to job {job_id} within {TRAIN_TIMEOUT}s")


def train_client_run_aggvfl(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None):
    """Feature/label party: align rows and share this party's own
    columns into the aggVFL training computation (Rep3 MPC - private).

    Same as train_client_run (aggVFLc), except the label party also
    secret-shares its own features here, not just the label. Internally
    reuses the same Private PSI logic as every other function here to
    independently re-derive the aligned row set.
    """
    return _run_job("train_client_run_aggvfl", matching_method, fuzzy_threshold, run_id)


def train_party_run_aggvfl(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, schema: dict = None):
    """Computing party: run this party's role in the Rep3 aggVFL
    training computation (fixed-aggregation vertical logistic
    regression; unlike aggVFLc, the label party contributes features
    too). This computing party never sees any feature, label, or
    prediction - only its own secret share of the computation.

    schema: this run's discovered circuit shape - None uses the
    known-working default shape.
    """
    return _run_job("train_party_run_aggvfl", matching_method, fuzzy_threshold, run_id, schema)
