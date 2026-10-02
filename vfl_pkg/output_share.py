"""Researcher output as separately encrypted aggregator output shares.

aggregator_output_share runs on each aggregator's node, like the training
partials: it hands the request to the host daemon over the bridge and
returns the daemon's result unchanged - either {"status": "complete",
"run_id", "share_index", "share_envelope"} or a plain failure status. The
envelope opens only with the researcher's private key.

share_delivery_central dispatches one such subtask per aggregator (the same
route central_train uses for its aggregator subtasks) and returns the three
envelopes as one result for the researcher's local tool
(daemons/researcher_reconstruct.py). It never sees a share in plaintext and
does not combine anything; it only reports whether all three arrived.
"""
import os
import uuid

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.util import info

from ._bridge_io import publish_job, wait_for_result

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")
ENVELOPE_RESULT_KEYS = {"status", "run_id", "share_index", "share_envelope"}


def researcher_output_check(recipient_key_sha256: str, run_id: str = None, timeout_s: int = 120):
    """Preflight for secure training, run on every data holder and computing
    party before any PSI or training task is dispatched: this host's own
    researcher-output configuration must exist, its signing key must match
    its certificate (and, on a computing party, its collector must be set
    up), and its configured researcher key must be the one the task names.
    Returns {"status": "complete", "researcher_output": "ready"} or a plain
    error status with a bare reason - nothing about the keys themselves."""
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "action": "researcher_output_check", "recipient_key_sha256": recipient_key_sha256}
    if run_id:
        job["run_id"] = run_id
    publish_job(JOBS_DIR, job)
    return wait_for_result(RESULTS_DIR, job_id, timeout_s, label="researcher output check")


def aggregator_output_share(run_id: str, mpc_port: int, architecture: str, algorithm: str, schema: dict,
                            feature_order: dict, recipient_key_sha256: str, timeout_s: int = 900):
    job_id = str(uuid.uuid4())
    publish_job(JOBS_DIR, {"job_id": job_id, "action": "output_share_party_run", "run_id": run_id,
                           "mpc_port": mpc_port, "architecture": architecture, "algorithm": algorithm,
                           "schema": schema, "feature_order": feature_order,
                           "recipient_key_sha256": recipient_key_sha256})
    info(f"Output share: waiting for run {run_id}")
    result = wait_for_result(RESULTS_DIR, job_id, timeout_s, label="output share")
    info(f"Output share: run {run_id} finished with status {result.get('status')}")
    return result


@algorithm_client
def share_delivery_central(client: AlgorithmClient, agg_org_ids: list, run_id: str, mpc_port: int,
                           architecture: str, algorithm: str, schema: dict, feature_order: dict,
                           recipient_key_sha256: str):
    if len(agg_org_ids) != 3 or len(set(agg_org_ids)) != 3:
        raise ValueError("exactly three distinct aggregator organizations are required")
    kwargs = {"run_id": run_id, "mpc_port": mpc_port, "architecture": architecture, "algorithm": algorithm,
              "schema": schema, "feature_order": feature_order, "recipient_key_sha256": recipient_key_sha256}
    tasks = {}
    for org_id in agg_org_ids:
        t = client.task.create(input_={"method": "aggregator_output_share", "kwargs": kwargs},
                               organizations=[org_id], name=f"output-share-{run_id[:8]}-agg-{org_id}")
        tasks[org_id] = t["id"]
    info(f"Output share central: {len(tasks)} aggregator subtasks submitted for run {run_id}")

    envelopes, aggregators = [], {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        r = res[0] if res else None
        if isinstance(r, dict) and r.get("status") == "complete" and set(r) == ENVELOPE_RESULT_KEYS \
                and r.get("run_id") == run_id and isinstance(r.get("share_envelope"), dict):
            envelopes.append(r["share_envelope"])
            aggregators[str(org_id)] = {"status": "complete", "share_index": r.get("share_index")}
        else:
            aggregators[str(org_id)] = {"status": (r or {}).get("status", "no result"),
                                        "reason": (r or {}).get("reason")}
    delivered = len(envelopes) == 3
    info(f"Output share central: run {run_id} {'delivered' if delivered else 'incomplete'}")
    out = {"status": "delivered" if delivered else "incomplete", "run_id": run_id, "aggregators": aggregators}
    if delivered:
        out["shares"] = envelopes
    return out
