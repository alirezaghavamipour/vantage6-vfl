from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


@algorithm_client
def central(client: AlgorithmClient, client_org_ids: list, agg_org_ids: list):
    """
    Orchestrate a full Rep3 PSI run in one submission: dispatch
    psi_client_share to each feature/label party and psi_party_run to
    each computing party, then collect and summarize the result.
    """
    info(f"Central: starting PSI run - clients={client_org_ids}, aggregators={agg_org_ids}")

    tasks = {}
    for org_id in client_org_ids:
        t = client.task.create(
            input_={"method": "psi_client_share", "kwargs": {}},
            organizations=[org_id],
            name=f"psi-client-{org_id}",
        )
        tasks[org_id] = t["id"]

    for org_id in agg_org_ids:
        t = client.task.create(
            input_={"method": "psi_party_run", "kwargs": {}},
            organizations=[org_id],
            name=f"psi-agg-{org_id}",
        )
        tasks[org_id] = t["id"]

    info(f"Central: all {len(tasks)} sub-tasks submitted, waiting for results...")

    results = {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        results[org_id] = res[0] if res else None

    intersection_size = None
    total_entities = None
    slots = None
    for org_id in agg_org_ids:
        r = results.get(org_id)
        if r and r.get("status") == "complete":
            slots = r.get("slots")
            total_entities = len(slots)
            intersection_size = sum(1 for v in slots.values() if v == 1)
            break

    info(f"Central: PSI complete - {intersection_size}/{total_entities} entities matched")

    return {
        "summary": [
            {"metric": "intersection_size", "value": intersection_size},
            {"metric": "total_entities", "value": total_entities},
        ],
        "intersection_size": intersection_size,
        "total_entities": total_entities,
        "slots": slots,
        "client_results": {org_id: results.get(org_id) for org_id in client_org_ids},
        "aggregator_results": {org_id: results.get(org_id) for org_id in agg_org_ids},
    }
