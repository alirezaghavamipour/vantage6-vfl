import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")


# The fuzzy method (oblivious blocking + banded edit distance + nickname
# canonicalization) does more MPC work than exact-hash matching - measured
# at ~14 minutes at this deployment's scale (833s, real measured run, not
# an estimate), vs seconds for exact. Timeout kept well above that.
TIMEOUT_BY_METHOD = {"exact": 120, "fuzzy": 1500}

# The edit-distance threshold shapes the MPC circuit itself (band width),
# so it can't be a runtime argument to an already-compiled program - only
# these thresholds have a pre-compiled circuit on the aggregators.
SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None, max_entities: int = None, debug: bool = False) -> dict:
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
    # Whether this party's own subtask result may include row-level
    # sensitive data (matched names). Independent of any debug filtering
    # central()/central_train() apply to their own top-level summary -
    # this subtask's result is separately queryable in vantage6.
    if debug:
        job["debug"] = True
    # run_id ties every job dispatched by one orchestrator call (central())
    # together across however many hosts they land on - job_id alone is
    # only unique to this one job, with nothing connecting it to its
    # siblings from the same run. Optional/None for direct/manual
    # invocation outside an orchestrator.
    if run_id:
        job["run_id"] = run_id
    # max_entities: public upper bound on any single party's raw
    # candidate row count (before matching) this run's PSI circuit
    # should be compiled for - None uses the computing party's own
    # default (320). Auto-computed by central()/central_train() from
    # schema discovery's reported row counts, not usually set by hand.
    if max_entities:
        job["max_entities"] = max_entities
    timeout = TIMEOUT_BY_METHOD.get(matching_method, 120)
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"PSI: submitted job {job_id} (run_id={run_id}, {action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(timeout):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"PSI: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"PSI: host daemon did not respond to job {job_id} within {timeout}s")


def psi_client_share(matching_method: str = "exact", fuzzy_threshold: int = 2,
                      run_id: str = None, max_entities: int = None, debug: bool = False):
    """Feature/label party: share local entity data into the MPC computation.

    matching_method: "exact" (hash-based exact match) or "fuzzy"
    (nickname-canonicalized, typo-tolerant edit-distance match).
    fuzzy_threshold: max character edits allowed (1, 2, or 3) when
    matching_method="fuzzy" - higher tolerates more typos but also risks
    matching different people with similar names. Ignored for "exact".
    run_id: shared identifier set by central() to correlate this job with
    the other jobs dispatched by the same orchestrated run, across every
    host they land on. Not meant to be set when calling this directly.
    max_entities: this run's row-count bound, if central()'s schema
    discovery raised it above the default - None uses this daemon's own
    local default (320).
    debug: if True, this party's own result includes the matched names/
    align keys it discovered. Off by default - those fields aren't used
    by central()'s own aggregation (it only needs intersection_size), so
    leaving this False keeps them out of this subtask's stored result
    too, not just out of central()'s filtered top-level summary.
    """
    return _run_job("psi_client_run", matching_method, fuzzy_threshold, run_id, max_entities, debug)


def psi_party_run(matching_method: str = "exact", fuzzy_threshold: int = 2,
                   run_id: str = None, max_entities: int = None):
    """Computing party: run this party's role in the Rep3 PSI computation.

    run_id: see psi_client_share - shared identifier set by central() for
    cross-host job correlation. max_entities: see psi_client_share.
    """
    return _run_job("psi_party_run", matching_method, fuzzy_threshold, run_id, max_entities)
