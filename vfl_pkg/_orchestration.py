"""Dispatching a round of subtasks and collecting its results per organization.

dispatch() creates one task per role (all organizations of the role in one
task) or, with group_by_role=False, one task per organization. collect()
waits for every run and keys each result by the organization that produced
it (run id -> organization from the run records), never by arrival order.
"""
import base64
import json
import time

from vantage6.algorithm.tools.util import info

ACTIVE_STATUSES = {"pending", "initializing", "active"}


def dispatch(client, method, kwargs, org_ids, name, group_by_role):
    """Returns {task_id: [org_id, ...]}."""
    org_ids = list(org_ids)
    if group_by_role:
        t = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids, name=name)
        return {t["id"]: org_ids}
    tasks = {}
    for org_id in org_ids:
        t = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=[org_id],
                               name=f"{name}-{org_id}")
        tasks[t["id"]] = [org_id]
    return tasks


def _decode(raw):
    if not raw:
        return None
    try:
        return json.loads(base64.b64decode(raw).decode())
    except Exception:
        return None


def collect(client, task_map, poll_interval=1.0, until=None, on_failure=None):
    """Waits for every run of every task in task_map and returns
    ({org_id: result}, stats). A failed run or a result whose status is not
    "complete" triggers on_failure() once; runs still unfinished at `until`
    are reported as timed out."""
    expected = {org for orgs in task_map.values() for org in orgs}
    results, failure_signalled, requests = {}, False, 0
    if not hasattr(client, "run"):
        # Clients without run listings (simple mock clients): one organization per task only.
        for task_id, orgs in task_map.items():
            if len(orgs) != 1:
                raise ValueError("grouped tasks need a client with run listings to map results to organizations")
            res = client.wait_for_results(task_id=task_id)
            results[orgs[0]] = (res[0] if res else None) or {"status": "error", "message": "no result"}
        if on_failure and any(r.get("status", "complete") != "complete" for r in results.values()):
            on_failure()
        return results, {"api_requests": None}
    while True:
        for task_id, orgs in task_map.items():
            if set(orgs) <= set(results):
                continue
            runs = client.run.from_task(task_id)
            requests += 1
            finished = {r["id"]: r for r in runs if r.get("status") not in ACTIVE_STATUSES
                        and r["organization"]["id"] not in results}
            if not finished:
                continue
            payloads = {r["id"]: _decode(r.get("result"))
                        for r in client._multi_page_request("result", params={"task_id": task_id})}
            requests += 1
            for run_id, run in finished.items():
                org = run["organization"]["id"]
                if org not in orgs:
                    continue
                result = payloads.get(run_id) if run.get("status") == "completed" else None
                if result is None:
                    result = {"status": "error", "message": f"subtask run {run.get('status')}"}
                results[org] = result
                if result.get("status", "complete") != "complete" and on_failure and not failure_signalled:
                    failure_signalled = True
                    info(f"subtask for organization {org} ended with status {result.get('status')}; cancelling run")
                    on_failure()
        if expected <= set(results):
            return results, {"api_requests": requests}
        if until is not None and time.time() > until:
            for org in expected - set(results):
                results[org] = {"status": "error", "message": "no result by the round deadline"}
            if on_failure and not failure_signalled:
                on_failure()
            return results, {"api_requests": requests}
        time.sleep(poll_interval)


def cancel_run(client, run_id, org_ids, group_by_role=True):
    """Asks every listed organization's daemon to stop all jobs of run_id."""
    return dispatch(client, "cancel_run", {"run_id": run_id}, org_ids, f"cancel-{run_id[:8]}", group_by_role)
