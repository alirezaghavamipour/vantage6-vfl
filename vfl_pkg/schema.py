import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

from . import _bridge_io
from ._bridge_io import publish_job, wait_for_result

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# A pure local CSV read (no MPC computation at all), so this should
# always come back in well under a second - generous timeout is just
# headroom, not an estimate of real work.
TIMEOUT = 60


def report_schema_run(database: str = None, run_id: str = None, include_row_count: bool = True,
                       capacity_mode: str = "manual", deadline: float = None, poll_interval: float = None):
    """Feature/label party: report this party's local row count and
    feature-column count for the given database, read directly from its
    own CSV (every column except full_name/target) - lets central_train
    learn the actual joint circuit shape (row-count bound, per-party
    feature counts) before dispatching training, instead of assuming a
    fixed dataset shape. The aggregators use this to decide whether
    their compiled training circuit needs regenerating for this run.

    database: which of this party's registered csv databases to report
    on - None means "heart_vfl". Resolved by central_train/central from
    the selected dataset and architecture (see datasets.py), not meant to
    be chosen by hand.
    run_id: shared identifier set by central_train to correlate this job
    with the other jobs dispatched by the same orchestrated run.
    include_row_count: Phase 2 Stage 3 - False for automatic capacity
    mode (psi_capacity="auto"), which discovers row counts exclusively
    through the private row-count-discovery MPC circuit instead. This
    party's n_rows is then omitted from the result entirely (not
    zeroed/redacted-in-place); n_features/client_id are unaffected.
    capacity_mode: bug #3 fix - "dynamic" (automatic capacity mode)
    makes this read the run's already-retained dataset snapshot instead
    of a fresh CSV read, so the reported feature count matches the exact
    rows counting/PSI/training all use for this run_id. Requires private
    row-count discovery (and therefore snapshot retention) to have
    already run for this run_id - see train_unified.py's/central.py's
    own dispatch ordering for automatic mode, which negotiates capacity
    BEFORE dispatching schema discovery for this reason.
    """
    _bridge_io.configure(poll_interval=poll_interval, deadline=deadline)
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "report_schema"}
    if database:
        job["database"] = database
    if run_id:
        job["run_id"] = run_id
    if not include_row_count:
        job["include_row_count"] = False
    # count-disclosure fix's own convention (see psi.py's _run_job).
    if capacity_mode != "manual":
        job["capacity_mode"] = capacity_mode
    publish_job(JOBS_DIR, job, timeout_s=TIMEOUT)
    info(f"Schema: submitted job {job_id} (run_id={run_id}, database={database}), "
         f"waiting for host daemon...")
    result = wait_for_result(RESULTS_DIR, job_id, TIMEOUT, label="Schema")
    info(f"Schema: job {job_id} completed with status {result.get('status')}")
    return result
