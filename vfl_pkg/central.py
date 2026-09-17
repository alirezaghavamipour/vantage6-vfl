from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


@algorithm_client
def central(client: AlgorithmClient, client_org_ids: list, agg_org_ids: list, debug: bool = False):
    """
    Orchestrate a full Rep3 PSI run in one submission: dispatch
    psi_client_share to each feature/label party and psi_party_run to
    each computing party, then collect and summarize the result.

    By default only the merged final answer is returned - not each
    party's individual raw result - to avoid unnecessarily exposing
    e.g. each client's own local dataset size. Set debug=True to also
    include the full per-party results, for auditing one specific run.
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

    # The aggregators never learn the answer - the MPC circuit sends each of
    # them only their own share of the output (sint.reveal_to_clients), and
    # each client independently reconstructs the plaintext locally from the
    # 3 shares it receives. So the final answer is read from the clients,
    # not the aggregators - and since every client reconstructs it
    # independently, we can cross-check that they all agree as extra
    # evidence of correctness (Rep3's guarantee assumes at most 1 of the 3
    # computing parties is dishonest).
    intersection_size = None
    total_entities = None
    slots = None
    client_answers = {}
    for org_id in client_org_ids:
        r = results.get(org_id)
        if r and r.get("status") == "complete":
            client_answers[org_id] = (r.get("intersection_size"), r.get("total_entities"))
            if intersection_size is None:
                intersection_size = r.get("intersection_size")
                total_entities = r.get("total_entities")
                slots = r.get("slots")

    clients_agree = len(set(client_answers.values())) <= 1 if client_answers else None

    info(
        f"Central: PSI complete - {intersection_size}/{total_entities} entities matched "
        f"(clients_agree={clients_agree})"
    )

    output = {
        "summary": [
            {"metric": "intersection_size", "value": intersection_size},
            {"metric": "total_entities", "value": total_entities},
            {"metric": "clients_agree", "value": clients_agree},
        ],
        "intersection_size": intersection_size,
        "total_entities": total_entities,
        "clients_agree": clients_agree,
        "slots": slots,
    }

    if debug:
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
