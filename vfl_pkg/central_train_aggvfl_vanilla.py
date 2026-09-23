from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


@algorithm_client
def central_train_aggvfl_vanilla(
    client: AlgorithmClient,
    feature_org_ids: list,
    label_org_id: int,
    agg_org_ids: list,
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    debug: bool = False,
):
    """
    NOT PRIVATE - a deliberately insecure baseline, provided only to
    compare against a future real (Rep3 MPC) aggVFL training function.
    Do not use this for any data that needs real protection.

    aggVFL = fixed-aggregation vertical FL where the active (label)
    party ALSO contributes its own features to the model - unlike
    aggVFLc, where the label party has no features at all. Otherwise
    identical in structure to 'Train aggVFLc model (vanilla, NOT
    PRIVATE)': dispatches vanilla_train_aggvfl_worker_run to each
    feature party and vanilla_train_aggvfl_coordinator_run to the label
    party, who coordinate training directly with each other over a
    plain socket (not MPC), plus psi_party_run to the computing parties
    for row alignment only (still the private Rep3 PSI circuit).

    What "not private" means concretely: every epoch, the label party
    learns each feature party's partial linear score in the clear, and
    every feature party learns (prediction - true label) in the clear -
    the standard Direct Label Inference leak this whole "vanilla" family
    of functions exists to illustrate, as a baseline for the real MPC
    versions.

    By default only the merged summary is returned. Set debug=True to
    also include the full per-party results, for auditing one specific
    run.
    """
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )

    info(f"Central (train aggVFL VANILLA - NOT PRIVATE): starting training run "
         f"(method={matching_method}, fuzzy_threshold={fuzzy_threshold}) - "
         f"features={feature_org_ids}, label={label_org_id}, aggregators={agg_org_ids}")

    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold}
    client_org_ids = list(feature_org_ids) + [label_org_id]

    tasks = {}
    for org_id in feature_org_ids:
        t = client.task.create(
            input_={"method": "vanilla_train_aggvfl_worker_run", "kwargs": kwargs},
            organizations=[org_id],
            name=f"vanilla-train-aggvfl-worker-{org_id}",
        )
        tasks[org_id] = t["id"]

    t = client.task.create(
        input_={"method": "vanilla_train_aggvfl_coordinator_run", "kwargs": kwargs},
        organizations=[label_org_id],
        name=f"vanilla-train-aggvfl-coordinator-{label_org_id}",
    )
    tasks[label_org_id] = t["id"]

    for org_id in agg_org_ids:
        t = client.task.create(
            input_={"method": "psi_party_run", "kwargs": kwargs},
            organizations=[org_id],
            name=f"vanilla-train-aggvfl-psi-agg-{org_id}",
        )
        tasks[org_id] = t["id"]

    info(f"Central (train aggVFL VANILLA): all {len(tasks)} sub-tasks submitted, waiting for results...")

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

    info(f"Central (train aggVFL VANILLA): training complete - aligned_count={aligned_count} "
         f"(predictions_agree={predictions_agree}, aggregators_ok={aggregators_ok})")

    output = {
        "summary": [
            {"metric": "privacy", "value": "NOT PRIVATE - plaintext baseline"},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "aligned_count", "value": aligned_count},
            {"metric": "predictions_agree", "value": predictions_agree},
            {"metric": "aggregators_ok", "value": aggregators_ok},
        ],
        "privacy": "NOT PRIVATE - plaintext baseline",
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
