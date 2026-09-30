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
    info(f"Vanilla splitVFL train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result = wait_for_result(RESULTS_DIR, job_id, VANILLA_TIMEOUT, label="Vanilla splitVFL train")
    info(f"Vanilla splitVFL train: job {job_id} completed with status {result.get('status')}")
    return result


def vanilla_train_splitvfl_bottom_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual"):
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
    return _run_job("vanilla_train_splitvfl_bottom_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode)


def vanilla_train_splitvfl_top_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, database_by_client_id: dict = None, max_entities: int = None,
                    approved_mapping_digest_by_client_id: dict = None, capacity_mode: str = "manual"):
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
    return _run_job("vanilla_train_splitvfl_top_run", matching_method, fuzzy_threshold, run_id,
                     database_by_client_id=database_by_client_id, max_entities=max_entities,
                     approved_mapping_digest_by_client_id=approved_mapping_digest_by_client_id,
                     capacity_mode=capacity_mode)
