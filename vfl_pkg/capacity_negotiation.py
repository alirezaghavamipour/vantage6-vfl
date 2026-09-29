"""Phase 2 Stage 3: automatic PSI capacity negotiation - task-submission
wrappers for the two new daemon-side steps that determine and prepare a
capacity WITHOUT a manual selection: private row-count discovery
(count_discovery_client_run / count_discovery_run) and the on-demand
build step (prepare_dynamic_capacity_run). Same shape as schema.py/
psi.py's own wrappers (a job dict, published atomically, then polled
for a result) - this file only adds the two NEW job shapes those don't
already cover.
"""
import os
import uuid

from vantage6.algorithm.tools.util import info

from ._bridge_io import publish_job, wait_for_result

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# Row-count discovery is a tiny, constant-size MPC computation (3 secret
# integers) - generous but not open-ended.
COUNT_DISCOVERY_TIMEOUT = 120
# Dynamic capacity compilation - same bound as fuzzy_experimental's own
# manual-capacity precompile step (EXPERIMENTAL_COMPILE_TIMEOUT in
# mpc_daemon_agg_v2.py), since it's the identical base circuit/flags,
# just parameterized differently. Client-side wait must exceed the
# daemon's own compile timeout so it's never the first thing to expire.
PREPARE_DYNAMIC_CAPACITY_TIMEOUT = 2500


def count_discovery_client_run(run_id: str = None, database_by_client_id: dict = None):
    """Feature/label party: retain this party's dataset snapshot and
    submit its row count as a secret share to the private row-count
    discovery circuit. Returns only what that circuit reveals - a
    collective validity flag and the maximum row count across all 3
    parties - never this or any other party's individual count.

    database_by_client_id: same convention as psi_client_share's own -
    keyed by CLIENT_ID (stringified), each daemon self-selects its own
    database entry. Needed so the RETAINED snapshot (later consumed by
    automatic-mode PSI, and matched against by training) is read from
    the same database a later training call for this party will use
    (same rationale as psi_client_share's own database_by_client_id).
    """
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "count_discovery_client_run"}
    if run_id:
        job["run_id"] = run_id
    job["database_by_client_id"] = database_by_client_id or {}
    publish_job(JOBS_DIR, job)
    info(f"CapacityNegotiation: submitted count_discovery_client_run job {job_id} "
         f"(run_id={run_id}), waiting for host daemon...")
    result = wait_for_result(RESULTS_DIR, job_id, COUNT_DISCOVERY_TIMEOUT, label="CapacityNegotiation")
    info(f"CapacityNegotiation: job {job_id} completed with status {result.get('status')}")
    return result


def count_discovery_run(run_id: str = None):
    """Computing party: run this party's role in the private row-count
    discovery circuit. Reveals nothing to this computing party - see
    private_max_discovery.mpc's own module docstring."""
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "count_discovery_run"}
    if run_id:
        job["run_id"] = run_id
    publish_job(JOBS_DIR, job)
    info(f"CapacityNegotiation: submitted count_discovery_run job {job_id} "
         f"(run_id={run_id}), waiting for host daemon...")
    result = wait_for_result(RESULTS_DIR, job_id, COUNT_DISCOVERY_TIMEOUT, label="CapacityNegotiation")
    info(f"CapacityNegotiation: job {job_id} completed with status {result.get('status')}")
    return result


def prepare_dynamic_capacity_run(run_id: str = None, max_entities: int = None, fuzzy_threshold: int = 2):
    """Computing party: compile (or confirm cached) the fuzzy_experimental
    circuit for max_entities - the ONLY place a dynamic capacity actually
    gets compiled, dispatched separately from and before psi_party_run
    (which is check-only for dynamic capacities). Returns this
    aggregator's own build fingerprint and complete-artifact bytecode
    hash - central_train's build-agreement barrier requires all 3
    aggregators' responses to match exactly before any psi_party_run is
    dispatched."""
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "prepare_dynamic_capacity_run", "fuzzy_threshold": fuzzy_threshold}
    if run_id:
        job["run_id"] = run_id
    if max_entities:
        job["max_entities"] = max_entities
    publish_job(JOBS_DIR, job)
    info(f"CapacityNegotiation: submitted prepare_dynamic_capacity_run job {job_id} "
         f"(run_id={run_id}, max_entities={max_entities}), waiting for host daemon...")
    result = wait_for_result(RESULTS_DIR, job_id, PREPARE_DYNAMIC_CAPACITY_TIMEOUT, label="CapacityNegotiation")
    info(f"CapacityNegotiation: job {job_id} completed with status {result.get('status')}")
    return result
