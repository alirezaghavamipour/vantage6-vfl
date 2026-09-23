from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


@algorithm_client
def central_train_splitvflc(
    client: AlgorithmClient,
    client_org_ids: list,
    agg_org_ids: list,
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    debug: bool = False,
):
    """
    Orchestrate a full splitVFLc training run in one submission: dispatch
    train_client_run_splitvflc to each feature/label party and
    train_party_run_splitvflc to each computing party, then collect and
    summarize the result.

    splitVFLc = trainable-module vertical FL where the active (label)
    party has no features of its own (same shape as aggVFLc), but the
    fixed sigmoid-over-a-linear-combination is replaced by a genuine
    small neural network trained entirely inside the Rep3 MPC
    computation: a Dense+ReLU hidden layer over the concatenated
    features, then a Dense+Sigmoid output layer. Unlike the vanilla
    (NOT PRIVATE) version of this function, the private version doesn't
    need to structurally separate each party's own "bottom model" -
    since nothing is ever visible outside the MPC computation regardless
    of internal layer wiring, the hidden layer is fully connected across
    all parties' combined features rather than split per party. This is
    a deliberate simplification specific to the MPC setting (the
    per-party split in the vanilla version exists only to avoid a
    plaintext party ever seeing another party's raw contribution, a
    concern that doesn't apply here).

    Like every other training function here, reuses the exact same
    Private PSI circuit internally to obliviously align rows before
    training - features and label live on different parties' machines
    and are combined only inside the Rep3 MPC computation, never in
    plaintext anywhere, not even on the computing parties.

    matching_method / fuzzy_threshold: same meaning as every other
    function here - see 'Run private PSI' for the full explanation.

    Predictions are revealed identically to every client party, so each
    one can independently verify the trained model - this function
    cross-checks that they all agree.

    By default only the merged summary is returned. Set debug=True to
    also include the full per-party results, for auditing one specific
    run.
    """
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )

    info(f"Central (train splitVFLc): starting training run (method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}) - clients={client_org_ids}, aggregators={agg_org_ids}")

    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold}

    tasks = {}
    for org_id in client_org_ids:
        t = client.task.create(
            input_={"method": "train_client_run_splitvflc", "kwargs": kwargs},
            organizations=[org_id],
            name=f"train-splitvflc-client-{org_id}",
        )
        tasks[org_id] = t["id"]

    for org_id in agg_org_ids:
        t = client.task.create(
            input_={"method": "train_party_run_splitvflc", "kwargs": kwargs},
            organizations=[org_id],
            name=f"train-splitvflc-agg-{org_id}",
        )
        tasks[org_id] = t["id"]

    info(f"Central (train splitVFLc): all {len(tasks)} sub-tasks submitted, waiting for results...")

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

    info(f"Central (train splitVFLc): training complete - aligned_count={aligned_count} "
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
