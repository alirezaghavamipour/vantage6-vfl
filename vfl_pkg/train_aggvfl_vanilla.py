import os
import uuid

from vantage6.algorithm.tools.util import info

from ._bridge_io import publish_job, wait_for_result

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

VANILLA_TIMEOUT = 1800

SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
             approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual") -> dict:
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
    # database_by_client_id: R06 fix - each client's own database
    # selection, keyed by CLIENT_ID, so PSI and training on this party
    # read the exact same file. Rows never travel through this job dict -
    # only the resolved database LABEL (a string).
    if database_by_client_id:
        job["database_by_client_id"] = database_by_client_id
    # max_entities: R11 fix - see train_vanilla.py's matching comment.
    if max_entities:
        job["max_entities"] = max_entities
    # approved_mapping_digest_by_client_id: k=1 Stage 3 gap-2 fix - see
    # train_vanilla.py's matching comment.
    if approved_mapping_digest_by_client_id:
        job["approved_mapping_digest_by_client_id"] = approved_mapping_digest_by_client_id
    # count-disclosure fix: see train.py's matching comment.
    if capacity_mode != "manual":
        job["capacity_mode"] = capacity_mode
    publish_job(JOBS_DIR, job)
    info(f"Vanilla aggVFL train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result = wait_for_result(RESULTS_DIR, job_id, VANILLA_TIMEOUT, label="Vanilla aggVFL train")
    info(f"Vanilla aggVFL train: job {job_id} completed with status {result.get('status')}")
    return result


def vanilla_train_aggvfl_worker_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual"):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) aggVFL training function.

    Feature party: re-aligns local rows (reusing the same Private PSI
    logic as every other function here), then runs its share of
    standard (non-MPC) vertical logistic regression - computes its own
    partial linear score and sends it to the label party IN THE CLEAR
    each epoch, receives the prediction error IN THE CLEAR back. Same
    protocol as vanilla_train_worker_run (aggVFLc); the only difference
    is which dataset/columns this party contributes, since aggVFL moves
    some columns from FP2 to the label party. Invoked internally by
    'central_train_aggvfl_vanilla', not meant to be run standalone.
    """
    return _run_job("vanilla_train_aggvfl_worker_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode)


def vanilla_train_aggvfl_coordinator_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual"):
    """NOT PRIVATE - deliberately insecure baseline for comparison
    against a future real (Rep3 MPC) aggVFL training function.

    Label party: unlike aggVFLc, this party now also holds its own
    features (not just the label) - it re-aligns local rows (reusing
    the same Private PSI logic), computes its own local partial linear
    score each epoch, combines it with the feature parties' partial
    scores received IN THE CLEAR, computes the sigmoid and prediction
    error using its own real labels, and broadcasts that error back to
    every feature party IN THE CLEAR. Invoked internally by
    'central_train_aggvfl_vanilla', not meant to be run standalone.
    """
    return _run_job("vanilla_train_aggvfl_coordinator_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode)
