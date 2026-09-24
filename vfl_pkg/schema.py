import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# A pure local CSV read (no MPC computation at all), so this should
# always come back in well under a second - generous timeout is just
# headroom, not an estimate of real work.
TIMEOUT = 60


def report_schema_run(database: str = None, run_id: str = None):
    """Feature/label party: report this party's local row count and
    feature-column count for the given database, read directly from its
    own CSV (every column except full_name/target) - lets central_train
    learn the actual joint circuit shape (row-count bound, per-party
    feature counts) before dispatching training, instead of assuming a
    fixed dataset shape. The aggregators use this to decide whether
    their compiled training circuit needs regenerating for this run.

    database: which of this party's registered databases to report on -
    "heart_vfl" (the default) or "heart_vfl_aggvfl" for architectures
    where the label party also contributes features. Set by
    central_train per party's role, not meant to be chosen by hand.
    run_id: shared identifier set by central_train to correlate this job
    with the other jobs dispatched by the same orchestrated run.
    """
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "report_schema"}
    if database:
        job["database"] = database
    if run_id:
        job["run_id"] = run_id
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Schema: submitted job {job_id} (run_id={run_id}, database={database}), "
         f"waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Schema: job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Schema: host daemon did not respond to job {job_id} within {TIMEOUT}s")
