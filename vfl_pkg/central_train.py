from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


@algorithm_client
def central_train_aggvflc(
    client: AlgorithmClient,
    client_org_ids: list,
    agg_org_ids: list,
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    debug: bool = False,
):
    """
    Orchestrate a full aggVFLc training run in one submission: dispatch
    train_client_run to each feature/label party and train_party_run to
    each computing party, then collect and summarize the result.

    aggVFLc = fixed-aggregation vertical FL, active (label) party has no
    features of its own - the simplest of the 4 VFL architectures. This
    is a separate algorithm from vantage6-vfl-psi, even though it reuses
    the exact same Private PSI circuit internally (each computing party
    re-runs row alignment before training, since no state carries over
    between separate task submissions) to obliviously align rows before
    training a vertical logistic regression model whose features and
    label live on different parties' machines and are combined only
    inside the Rep3 MPC computation - never in plaintext anywhere, not
    even on the computing parties.

    matching_method: "exact" (hash-based exact name match - fast, but a
    single typo or nickname means that row is left out of training) or
    "fuzzy" (nickname-canonicalized + bounded edit-distance match,
    tolerant of typos and common nicknames - far more MPC work, expect
    tens of minutes at this deployment's scale). Fuzzy-matched rows are
    aligned across parties using a shared handle the PSI circuit itself
    computes and reveals for this purpose (not any party's real local
    row index) - exact match doesn't need this, since identical matched
    name strings already sort consistently across parties on their own.

    fuzzy_threshold: only used when matching_method="fuzzy" - see
    vantage6-vfl-psi's central() for the precision/recall tradeoff this
    controls. Only 1, 2, or 3 are supported (pre-compiled circuits).

    Predictions are revealed identically to every client party (not
    just the label party), so each one can independently verify the
    trained model - this function cross-checks that they all agree.

    By default only the merged summary is returned. Set debug=True to
    also include the full per-party results, for auditing one specific
    run.
    """
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )

    info(f"Central (train aggVFLc): starting training run (method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}) - clients={client_org_ids}, aggregators={agg_org_ids}")

    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold}

    tasks = {}
    for org_id in client_org_ids:
        t = client.task.create(
            input_={"method": "train_client_run", "kwargs": kwargs},
            organizations=[org_id],
            name=f"train-client-{org_id}",
        )
        tasks[org_id] = t["id"]

    for org_id in agg_org_ids:
        t = client.task.create(
            input_={"method": "train_party_run", "kwargs": kwargs},
            organizations=[org_id],
            name=f"train-agg-{org_id}",
        )
        tasks[org_id] = t["id"]

    info(f"Central (train aggVFLc): all {len(tasks)} sub-tasks submitted, waiting for results...")

    results = {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        results[org_id] = res[0] if res else None

    aligned_count = None
    predictions_by_org = {}
    for org_id in client_org_ids:
        r = results.get(org_id)
        if r and r.get("status") == "complete":
            predictions_by_org[org_id] = r.get("predictions")
            if aligned_count is None:
                aligned_count = r.get("aligned_count")

    predictions_agree = (
        len({tuple(p) for p in predictions_by_org.values() if p is not None}) <= 1
        if predictions_by_org else None
    )

    aggregators_ok = all(
        results.get(org_id) and results.get(org_id).get("status") == "complete"
        for org_id in agg_org_ids
    )

    info(f"Central (train aggVFLc): training complete - aligned_count={aligned_count} "
         f"(predictions_agree={predictions_agree}, aggregators_ok={aggregators_ok})")

    output = {
        "summary": [
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "aligned_count", "value": aligned_count},
            {"metric": "predictions_agree", "value": predictions_agree},
            {"metric": "aggregators_ok", "value": aggregators_ok},
        ],
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
        "aligned_count": aligned_count,
        "predictions_agree": predictions_agree,
        "aggregators_ok": aggregators_ok,
        "predictions": next(iter(predictions_by_org.values()), None),
    }

    if debug:
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
