import uuid

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)

# Maps (architecture, privacy_mode) to the underlying daemon actions to
# dispatch. "secure" always uses one uniform action for every client
# (feature and label parties are symmetric from the MPC circuit's point
# of view - the daemon itself figures out each party's role from its own
# CLIENT_ID). "non_secure" needs an asymmetric worker/coordinator split
# instead, since the label party plays a structurally different role
# (the plaintext coordinator every feature party connects to directly).
_ARCHITECTURES = {
    "aggVFLc": {
        "secure": {
            "client_action": "train_client_run",
            "agg_action": "train_party_run",
        },
        "non_secure": {
            "worker_action": "vanilla_train_worker_run",
            "coordinator_action": "vanilla_train_coordinator_run",
        },
    },
    "aggVFL": {
        "secure": {
            "client_action": "train_client_run_aggvfl",
            "agg_action": "train_party_run_aggvfl",
        },
        "non_secure": {
            "worker_action": "vanilla_train_aggvfl_worker_run",
            "coordinator_action": "vanilla_train_aggvfl_coordinator_run",
        },
    },
    "splitVFLc": {
        "secure": {
            "client_action": "train_client_run_splitvflc",
            "agg_action": "train_party_run_splitvflc",
        },
        "non_secure": {
            "worker_action": "vanilla_train_splitvflc_bottom_run",
            "coordinator_action": "vanilla_train_splitvflc_top_run",
        },
    },
    "splitVFL": {
        "secure": {
            "client_action": "train_client_run_splitvfl",
            "agg_action": "train_party_run_splitvfl",
        },
        "non_secure": {
            "worker_action": "vanilla_train_splitvfl_bottom_run",
            "coordinator_action": "vanilla_train_splitvfl_top_run",
        },
    },
}


@algorithm_client
def central_train(
    client: AlgorithmClient,
    feature_org_ids: list,
    label_org_id: int,
    agg_org_ids: list,
    architecture: str = "aggVFLc",
    privacy_mode: str = "secure",
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    debug: bool = False,
):
    """
    Train a vertical federated learning model - pick which of the 4
    architectures and whether to run it privately (Rep3 MPC) or as a
    deliberately insecure plaintext baseline for comparison.

    architecture:
      - 'aggVFLc': fixed-aggregation logistic regression, the label
        party has no features of its own (the simplest architecture).
      - 'aggVFL': same fixed aggregation, but the label party ALSO
        contributes its own features to the model.
      - 'splitVFLc': trainable-module SplitNN (a real small neural
        network - Dense+ReLU hidden layer, Dense+Sigmoid output),
        label party has no features of its own.
      - 'splitVFL': trainable-module SplitNN, label party also
        contributes its own features.

    privacy_mode:
      - 'secure': the real thing - all training happens inside a Rep3
        MPC computation. No party (including the computing parties)
        ever sees another party's features, label, or any intermediate
        value in plaintext.
      - 'non_secure': a deliberately insecure plaintext baseline, for
        comparing accuracy/speed against the secure version only - do
        not use this for any data that needs real protection. The
        label party coordinates directly with each feature party over
        a plain socket; for aggVFLc/aggVFL this reveals a single
        partial score and error value per epoch, for splitVFLc/splitVFL
        it reveals a full embedding vector and gradient vector per
        epoch (a stronger, well-documented "feature leakage" signal).

    Row alignment always uses the same private Rep3 PSI circuit
    (dispatched to agg_org_ids), regardless of privacy_mode - only the
    training step itself changes.

    matching_method / fuzzy_threshold: see 'Run private PSI' for the
    full explanation of exact vs. fuzzy entity matching.

    Predictions are revealed identically to every client party (feature
    and label parties alike), so each one can independently verify the
    trained model - this function cross-checks that they all agree.
    That reveal happens regardless of this function's own arguments; it
    is a property of the training circuit itself, not of this
    orchestrator.

    By default this function itself returns only a summary (aligned row
    count, whether the parties' predictions agreed, whether the
    computing parties completed successfully) - not the raw per-row
    predictions themselves, since the task submitter is not necessarily
    one of the data-holding organizations and has no inherent need to
    see every row's predicted label. Set debug=True to also include the
    full per-party results (including the raw predictions), for
    auditing one specific run.
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"architecture={architecture!r} is not supported "
            f"(choose one of {sorted(_ARCHITECTURES)})"
        )
    if privacy_mode not in ("secure", "non_secure"):
        raise ValueError(
            f"privacy_mode={privacy_mode!r} is not supported "
            f"(choose 'secure' or 'non_secure')"
        )
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )

    spec = _ARCHITECTURES[architecture][privacy_mode]
    # Ties every job this run dispatches - across every feature, label,
    # and computing party host - back to this one orchestrated run, so
    # anyone debugging the daemon-bridge job files on disk (which have
    # no other way to tell which host's job belongs to which run) can
    # find every piece of it. Also logged below and returned in the
    # summary.
    run_id = str(uuid.uuid4())
    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold, "run_id": run_id}
    client_org_ids = list(feature_org_ids) + [label_org_id]

    info(f"Central (train {architecture}, {privacy_mode}): starting training run "
         f"(run_id={run_id}, method={matching_method}, fuzzy_threshold={fuzzy_threshold}) - "
         f"features={feature_org_ids}, label={label_org_id}, aggregators={agg_org_ids}")

    tasks = {}
    if privacy_mode == "secure":
        for org_id in client_org_ids:
            t = client.task.create(
                input_={"method": spec["client_action"], "kwargs": kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-client-{org_id}",
            )
            tasks[org_id] = t["id"]
        for org_id in agg_org_ids:
            t = client.task.create(
                input_={"method": spec["agg_action"], "kwargs": kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-agg-{org_id}",
            )
            tasks[org_id] = t["id"]
    else:
        for org_id in feature_org_ids:
            t = client.task.create(
                input_={"method": spec["worker_action"], "kwargs": kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-worker-{org_id}",
            )
            tasks[org_id] = t["id"]
        t = client.task.create(
            input_={"method": spec["coordinator_action"], "kwargs": kwargs},
            organizations=[label_org_id],
            name=f"train-{architecture}-coordinator-{label_org_id}",
        )
        tasks[label_org_id] = t["id"]
        # Row alignment still needs the real Rep3 PSI circuit even in
        # non_secure mode - only the training math itself skips MPC.
        for org_id in agg_org_ids:
            t = client.task.create(
                input_={"method": "psi_party_run", "kwargs": kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-psi-agg-{org_id}",
            )
            tasks[org_id] = t["id"]

    info(f"Central (train {architecture}, {privacy_mode}): all {len(tasks)} sub-tasks submitted, waiting for results...")

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

    info(f"Central (train {architecture}, {privacy_mode}): training complete - aligned_count={aligned_count} "
         f"(predictions_agree={predictions_agree}, aggregators_ok={aggregators_ok})")

    output = {
        "summary": [
            {"metric": "run_id", "value": run_id},
            {"metric": "architecture", "value": architecture},
            {"metric": "privacy_mode", "value": privacy_mode},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "aligned_count", "value": aligned_count},
            {"metric": "predictions_agree", "value": predictions_agree},
            {"metric": "aggregators_ok", "value": aggregators_ok},
        ],
        "run_id": run_id,
        "architecture": architecture,
        "privacy_mode": privacy_mode,
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
        "aligned_count": aligned_count,
        "predictions_agree": predictions_agree,
        "aggregators_ok": aggregators_ok,
    }

    if debug:
        output["predictions"] = next(iter(predictions_by_org.values()), None)
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
