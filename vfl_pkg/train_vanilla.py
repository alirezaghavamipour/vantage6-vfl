import os
import uuid

from vantage6.algorithm.tools.util import info

from ._bridge_io import publish_job, wait_for_result

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
             run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
             approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual",
             algorithm: str = "logistic") -> dict:
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
    # max_entities: R11 fix - PSI's own row-count bound, now discovered
    # and forwarded for non_secure runs the same way secure mode already
    # gets it (see central_train) - without this, vanilla training's PSI
    # step always fell back to each daemon's own local default bound.
    if max_entities:
        job["max_entities"] = max_entities
    # approved_mapping_digest_by_client_id: k=1 Stage 3 gap-2 fix - this
    # was never forwarded for non_secure/vanilla training at all (only
    # the secure path's train.py had it), even though the underlying
    # daemon function (run_vanilla_train) already accepts and uses it.
    # Same convention as train.py's own _run_job: the FULL dict travels
    # in the job; the receiving daemon picks its own entry by CLIENT_ID.
    if approved_mapping_digest_by_client_id:
        job["approved_mapping_digest_by_client_id"] = approved_mapping_digest_by_client_id
    # count-disclosure fix: see train.py's matching comment.
    if capacity_mode != "manual":
        job["capacity_mode"] = capacity_mode
    # algorithm: now implemented for non_secure/vanilla training too
    # (previously rejected by central_train before reaching here) -
    # same omit-the-default convention as capacity_mode above; the
    # receiving daemon's own job.get("algorithm", "logistic") already
    # treats an absent key as the existing default, so existing
    # callers that never pass this at all keep getting logistic
    # exactly as before.
    if algorithm != "logistic":
        job["algorithm"] = algorithm
    publish_job(JOBS_DIR, job)
    info(f"Vanilla train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result = wait_for_result(RESULTS_DIR, job_id, VANILLA_TIMEOUT, label="Vanilla train")
    info(f"Vanilla train: job {job_id} completed with status {result.get('status')}")
    return result


def vanilla_train_worker_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual",
                    algorithm: str = "logistic"):
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
    return _run_job("vanilla_train_worker_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode, algorithm=algorithm)


def vanilla_train_coordinator_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual",
                    algorithm: str = "logistic"):
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
    return _run_job("vanilla_train_coordinator_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode, algorithm=algorithm)
