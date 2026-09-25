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
# on transient failures (240s each) on the training step, on top of the
# PSI-alignment step it always runs first - fuzzy alignment alone can
# take up to the ~3600s the aggregators allow it (see vantage6-vfl-psi's
# own fuzzy timeout). Give the vantage6 task layer enough headroom to
# never time out before the daemon's own worst case is exhausted.
TRAIN_TIMEOUT = 5000

SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None, schema: dict = None, algorithm: str = None,
             n_samples_bound: int = None, max_entities: int = None) -> dict:
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
    # run_id ties every job dispatched by one orchestrator call (central_train())
    # together across however many hosts they land on - job_id alone is only
    # unique to this one job. Optional/None for direct/manual invocation.
    if run_id:
        job["run_id"] = run_id
    # schema: this run's actual circuit shape (row-count bound, per-party
    # feature counts, epochs), discovered by central_train from the real
    # data instead of assumed fixed - lets the computing party's
    # ensure_circuit_compiled regenerate/recompile only when it actually
    # differs from what's already compiled. None falls back to today's
    # known-working 171-row/13-feature/200-epoch shape.
    if schema:
        job["schema"] = schema
    # algorithm: "logistic" (default) or "linear" - which circuit variant
    # this feature/label party should encode its data for.
    if algorithm:
        job["algorithm"] = algorithm
    # n_samples_bound: the SAME row-count bound sent to the computing
    # parties via schema["n_samples"] above, but sent directly to this
    # feature/label party too, since it independently needs to know how
    # many padding rows to add - without this, raising schema["n_samples"]
    # for the computing parties alone would leave this party still padding
    # to its own old default, causing a data-length mismatch.
    if n_samples_bound:
        job["n_samples_bound"] = n_samples_bound
    # max_entities: PSI's own row-count bound (raw candidate rows before
    # matching, separate from n_samples_bound which bounds the matched
    # intersection) - see vfl_pkg/psi.py's psi_client_share.
    if max_entities:
        job["max_entities"] = max_entities
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Train: submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

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


def train_client_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                      run_id: str = None, algorithm: str = None, n_samples_bound: int = None,
                      max_entities: int = None):
    """Feature/label party: align rows and share this party's own columns
    into the aggVFLc training computation.

    Internally reuses the same Private PSI logic as vantage6-vfl-psi to
    independently re-derive the aligned row set (no state is shared
    between separate task submissions), then secret-shares this party's
    own columns for those rows - features for a feature party, the label
    for the label party - so the model trains on data that is never
    combined in plaintext anywhere.

    matching_method: "exact" (hash-based exact name match) or "fuzzy"
    (nickname-canonicalized, typo-tolerant edit-distance match - see
    vantage6-vfl-psi for details). fuzzy_threshold: 1, 2, or 3 - only
    used when matching_method="fuzzy".
    run_id: shared identifier set by central_train to correlate this job
    with the other jobs dispatched by the same orchestrated run.
    algorithm: "logistic" (default) or "linear" - see central_train.
    n_samples_bound: this run's row-count bound, if central_train's
    schema discovery raised it above the default - None uses this
    daemon's own local default.
    """
    return _run_job("train_client_run", matching_method, fuzzy_threshold, run_id,
                     algorithm=algorithm, n_samples_bound=n_samples_bound, max_entities=max_entities)


def train_party_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                     run_id: str = None, schema: dict = None, max_entities: int = None):
    """Computing party: run this party's role in the Rep3 aggVFLc
    training computation (fixed-aggregation vertical logistic
    regression; the label party contributes no features of its own).
    This computing party never sees any feature, label, or prediction -
    only its own secret share of the computation.

    run_id: see train_client_run. schema: this run's discovered circuit
    shape (see central_train's schema-discovery step) - None uses the
    known-working default shape. max_entities: PSI's own row-count
    bound - None uses this daemon's own local default.
    """
    return _run_job("train_party_run", matching_method, fuzzy_threshold, run_id, schema, max_entities=max_entities)
