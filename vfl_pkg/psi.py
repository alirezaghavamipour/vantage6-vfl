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
    timeout = TIMEOUT_BY_METHOD.get(matching_method, 120)
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"PSI: submitted job {job_id} ({action}, method={matching_method}, "
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


def psi_client_share(matching_method: str = "exact", fuzzy_threshold: int = 2):
    """Feature/label party: share local entity data into the MPC computation.

    matching_method: "exact" (hash-based exact match) or "fuzzy"
    (nickname-canonicalized, typo-tolerant edit-distance match).
    fuzzy_threshold: max character edits allowed (1, 2, or 3) when
    matching_method="fuzzy" - higher tolerates more typos but also risks
    matching different people with similar names. Ignored for "exact".
    """
    return _run_job("psi_client_run", matching_method, fuzzy_threshold)


def psi_party_run(matching_method: str = "exact", fuzzy_threshold: int = 2):
    """Computing party: run this party's role in the Rep3 PSI computation."""
    return _run_job("psi_party_run", matching_method, fuzzy_threshold)
