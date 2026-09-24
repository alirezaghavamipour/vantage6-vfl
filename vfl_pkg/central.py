import uuid

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


@algorithm_client
def central(
    client: AlgorithmClient,
    client_org_ids: list,
    agg_org_ids: list,
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    debug: bool = False,
):
    """
    Orchestrate a full Rep3 PSI run in one submission: dispatch
    psi_client_share to each feature/label party and psi_party_run to
    each computing party, then collect and summarize the result.

    matching_method: "exact" (hash-based exact match on the full name -
    fast, but a single typo or nickname produces no match at all) or
    "fuzzy" (nickname-canonicalized + bounded edit-distance matching,
    tolerant of typos and common nicknames - far more MPC work, measured
    at ~14 minutes at this deployment's scale vs seconds for exact).

    fuzzy_threshold: only used when matching_method="fuzzy". Max number
    of character edits (insert/delete/substitute) allowed for a name to
    still count as a match - 1, 2, or 3. This is a precision/recall
    tradeoff, not a bug to be tuned away: a higher threshold catches more
    real typos but also risks merging two different people who happen to
    have similar names (e.g. "Martin"/"Martinez" are 2 edits apart).
    Lower threshold = fewer such false positives, but may miss some
    genuine typos (e.g. a single transposed pair of letters is 2 edits
    under standard edit distance, not 1). Default 2. The threshold shapes
    the MPC circuit itself, so only pre-compiled values are accepted.

    By default only the merged final answer is returned - not each
    party's individual raw result - to avoid unnecessarily exposing
    e.g. each client's own local dataset size. Set debug=True to also
    include the full per-party results, for auditing one specific run.
    """
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )

    # Ties every job this run dispatches - across all client and
    # aggregator hosts - back to this one orchestrated run, so anyone
    # debugging the daemon-bridge job files on disk (which have no other
    # way to tell which host's job belongs to which run) can find every
    # piece of it. Also logged below and returned in the summary.
    run_id = str(uuid.uuid4())

    info(f"Central: starting PSI run (run_id={run_id}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}) - clients={client_org_ids}, aggregators={agg_org_ids}")

    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold, "run_id": run_id}

    tasks = {}
    for org_id in client_org_ids:
        t = client.task.create(
            input_={"method": "psi_client_share", "kwargs": kwargs},
            organizations=[org_id],
            name=f"psi-client-{org_id}",
        )
        tasks[org_id] = t["id"]

    for org_id in agg_org_ids:
        t = client.task.create(
            input_={"method": "psi_party_run", "kwargs": kwargs},
            organizations=[org_id],
            name=f"psi-agg-{org_id}",
        )
        tasks[org_id] = t["id"]

    info(f"Central: all {len(tasks)} sub-tasks submitted, waiting for results...")

    results = {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        results[org_id] = res[0] if res else None

    # The aggregators never learn the answer - the MPC circuit obliviously
    # sorts all parties' (hashed) identifier values, finds 3-way matches,
    # and sends each aggregator only its own share of each client's
    # per-entry match flags (sint.reveal_to_clients). Each client
    # reconstructs its own match flags locally from the 3 shares it
    # receives and derives its own intersection_size - real entity values
    # are never revealed to any computing party, and no fixed/known entity
    # universe is assumed (matching values can be arbitrary identifiers).
    # Since all 3 clients independently compute the size of the same
    # underlying set, we cross-check that they agree as extra evidence of
    # correctness (Rep3's guarantee assumes at most 1 of the 3 computing
    # parties is dishonest).
    intersection_size = None
    client_answers = {}
    for org_id in client_org_ids:
        r = results.get(org_id)
        if r and r.get("status") == "complete":
            client_answers[org_id] = r.get("intersection_size")
            if intersection_size is None:
                intersection_size = r.get("intersection_size")

    clients_agree = len(set(client_answers.values())) <= 1 if client_answers else None

    info(f"Central: PSI complete - intersection_size={intersection_size} (clients_agree={clients_agree})")

    output = {
        "summary": [
            {"metric": "run_id", "value": run_id},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "intersection_size", "value": intersection_size},
            {"metric": "clients_agree", "value": clients_agree},
        ],
        "run_id": run_id,
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
        "intersection_size": intersection_size,
        "clients_agree": clients_agree,
    }

    if debug:
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
