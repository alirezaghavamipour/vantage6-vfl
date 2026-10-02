"""Researcher output node: collect the MPC release through this node's host
daemon (daemons/researcher_daemon.py) and return it as this task's result.

The result is either {"status": "delivered", "run_id", "envelope"} - an
envelope only the researcher's private key opens - or a plain status
("invalid", "failed", "error"). No model value is ever logged or returned
in plaintext.
"""
import os
import uuid

from vantage6.algorithm.tools.util import info

from ._bridge_io import publish_job, wait_for_result

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")


def researcher_receive(run_id: str, mpc_port: int, n_values: int, timeout_s: int = 900):
    job_id = str(uuid.uuid4())
    publish_job(JOBS_DIR, {"job_id": job_id, "action": "researcher_receive",
                           "run_id": run_id, "mpc_port": mpc_port, "n_values": n_values})
    info(f"Researcher output: waiting for run {run_id}")
    result = wait_for_result(RESULTS_DIR, job_id, timeout_s, label="researcher output")
    info(f"Researcher output: run {run_id} finished with status {result.get('status')}")
    return result
