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
    n_samples: int = None,
    algorithm: str = "logistic",
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

    n_samples (secure mode only, advanced): the training circuits are
    compiled for a fixed row-count BOUND, not an exact count - the true
    matched intersection can be anything up to that bound, handled via
    row-padding and a mask so padding rows never affect training (the
    default bound is 200, comfortably above today's known ~171-row
    intersection, so most training runs never need to touch this
    argument or trigger a recompile at all). Set n_samples if your
    dataset's true intersection could exceed 200: this run's shared bound
    is sent to every party that needs it - the computing parties (who
    recompile their circuit for it) and every feature/label party (who
    use it to correctly shape their padded data) - from this one
    argument, no separate manual redeployment required.

    Note: PSI has its own, separate row-count bound (a public upper
    bound on any party's RAW candidate row count before matching, as
    opposed to n_samples above which bounds the MATCHED intersection).
    Unlike n_samples, PSI's bound is fully automatic - there is no
    argument for it here, because it's always safely computable from
    each party's real row count (discovered the same way as the
    feature counts below) without needing PSI to run first the way the
    matched count does.

    algorithm (secure mode only):
      - 'logistic' (default): binary classification - the label must be
        an exact 0/1 value, predictions are 0/1 classifications.
      - 'linear': regression - the label is a continuous value,
        normalized and fixed-point encoded the same way every feature
        column already is. Predictions come back as NORMALIZED values
        (not yet scaled to the label's real units) so every party
        reports an identical number - only the label party's own
        result also includes 'label_max', since it's the only party
        that ever learns the label's true scale; multiply a normalized
        prediction by label_max to get the value in real units. Not yet
        available for privacy_mode='non_secure'.

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
    if algorithm not in ("logistic", "linear"):
        raise ValueError(
            f"algorithm={algorithm!r} is not supported "
            f"(choose 'logistic' or 'linear')"
        )
    if algorithm == "linear" and privacy_mode == "non_secure":
        raise ValueError(
            "algorithm='linear' is only implemented for privacy_mode='secure' "
            "so far - the non_secure vanilla baseline is still logistic-only"
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
    label_has_features = architecture in ("aggVFL", "splitVFL")

    info(f"Central (train {architecture}, {privacy_mode}): starting training run "
         f"(run_id={run_id}, method={matching_method}, fuzzy_threshold={fuzzy_threshold}) - "
         f"features={feature_org_ids}, label={label_org_id}, aggregators={agg_org_ids}")

    # Schema discovery: before dispatching training, ask each party to
    # report its own row count and feature-column count (read straight
    # from its CSV) - lets the aggregators learn this run's actual
    # circuit shape instead of assuming today's fixed 171-row/13-feature
    # dataset, and recompile their training circuit only when the shape
    # actually differs from what's already compiled (see
    # ensure_circuit_compiled on the aggregator daemon). Only meaningful
    # for the secure/MPC path - the non_secure vanilla path already
    # discovers its own columns locally at training time and has no
    # circuit to recompile.
    agg_kwargs = dict(kwargs)
    if privacy_mode == "secure":
        fp1_org, fp2_org = feature_org_ids[0], feature_org_ids[1]
        # The label party's own dataset only carries real feature columns
        # in the "heart_vfl_aggvfl" database (aggVFL/splitVFL); the plain
        # "heart_vfl" database gives it target+full_name only, which
        # report_schema correctly reports as 0 features.
        fp2_database = "heart_vfl_aggvfl" if label_has_features else "heart_vfl"
        lp_database = "heart_vfl_aggvfl" if label_has_features else "heart_vfl"

        schema_tasks = {
            fp1_org: client.task.create(
                input_={"method": "report_schema_run", "kwargs": {"database": "heart_vfl", "run_id": run_id}},
                organizations=[fp1_org], name=f"schema-{architecture}-{fp1_org}",
            )["id"],
            fp2_org: client.task.create(
                input_={"method": "report_schema_run", "kwargs": {"database": fp2_database, "run_id": run_id}},
                organizations=[fp2_org], name=f"schema-{architecture}-{fp2_org}",
            )["id"],
            label_org_id: client.task.create(
                input_={"method": "report_schema_run", "kwargs": {"database": lp_database, "run_id": run_id}},
                organizations=[label_org_id], name=f"schema-{architecture}-{label_org_id}",
            )["id"],
        }
        schema_results = {org_id: client.wait_for_results(task_id=task_id)[0]
                           for org_id, task_id in schema_tasks.items()}
        n_feat_a = schema_results[fp1_org]["n_features"]
        n_feat_b = schema_results[fp2_org]["n_features"]
        n_feat_c = schema_results[label_org_id]["n_features"] if label_has_features else 0
        # PSI's own row-count bound (a public upper bound on any single
        # party's RAW candidate row count, before matching - separate
        # from n_samples above, which bounds the MATCHED intersection
        # and can't be safely auto-computed the same way). This one CAN
        # be safely auto-computed: raw row counts are already known from
        # the schema discovery just above, with no chicken-and-egg
        # problem (unlike the matched count, raw counts don't need PSI
        # to run first). Rounded up to the next multiple of 50 with at
        # least 50 rows of headroom, so the bound never lands exactly on
        # any one party's true count.
        raw_row_counts = [schema_results[fp1_org]["n_rows"], schema_results[fp2_org]["n_rows"],
                           schema_results[label_org_id]["n_rows"]]
        max_entities = ((max(raw_row_counts) // 50) + 2) * 50
        agg_kwargs["max_entities"] = max_entities
        info(f"Central (train {architecture}, {privacy_mode}): discovered raw row counts "
             f"{raw_row_counts}, using PSI bound max_entities={max_entities}")
        # Feature counts are safe to auto-discover: each party's CSV
        # physically holds only its own columns, so "how many" is a
        # stable fact about the data regardless of which rows end up
        # matched. Row count is NOT safe to guess the same way - without
        # row-padding/masking (not yet built), the compiled circuit's row
        # count must equal the TRUE PSI-matched intersection size
        # exactly, which is smaller than any single party's raw row count
        # (PSI's whole job is finding that shared subset) and can only be
        # known by actually running PSI. So n_samples is an explicit
        # argument the caller supplies (e.g. from a prior 'Run private
        # PSI' call's reported intersection_size) rather than guessed
        # here - omitting it keeps today's proven default shape.
        # The schema is ALWAYS sent when discovery ran - feature counts
        # are always safe to auto-discover and forward (each party's CSV
        # only ever holds its own columns), so they must never be
        # silently dropped just because the row-count bound wasn't also
        # overridden. Only n_samples itself falls back to the computing
        # parties' current default (200) when not explicitly given -
        # that field alone needs the caller's explicit opt-in, since it
        # can't be safely auto-computed (see the comment above). Sending
        # a schema that matches what's already compiled is a no-op for
        # ensure_circuit_compiled (fingerprint match -> skip recompile),
        # so this doesn't add any cost for the common case where the
        # dataset's shape hasn't changed.
        schema = {
            "n_samples": n_samples if n_samples is not None else 200,
            "n_feat_a": n_feat_a, "n_feat_b": n_feat_b, "n_feat_c": n_feat_c,
            "n_epochs": 200, "algorithm": algorithm,
        }
        agg_kwargs["schema"] = schema
        info(f"Central (train {architecture}, {privacy_mode}): discovered schema {schema}")

    # algorithm/n_samples_bound are only valid kwargs for the secure
    # client actions - the non_secure vanilla worker/coordinator
    # functions don't accept them (algorithm='linear' is already
    # rejected together with privacy_mode='non_secure' above).
    # n_samples_bound is sent to the feature/label parties directly
    # (not just to the computing parties via agg_kwargs["schema"]),
    # since each of them independently needs to know how many padding
    # rows to add - without this, raising the bound for the computing
    # parties alone would leave every feature/label party still padding
    # to their own unchanged local default, causing a data-length
    # mismatch (the exact bug the schema-only version of this had).
    if privacy_mode == "secure":
        client_kwargs = dict(kwargs, algorithm=algorithm, max_entities=max_entities)
        if n_samples is not None:
            client_kwargs["n_samples_bound"] = n_samples
    else:
        client_kwargs = kwargs

    tasks = {}
    if privacy_mode == "secure":
        for org_id in client_org_ids:
            t = client.task.create(
                input_={"method": spec["client_action"], "kwargs": client_kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-client-{org_id}",
            )
            tasks[org_id] = t["id"]
        for org_id in agg_org_ids:
            t = client.task.create(
                input_={"method": spec["agg_action"], "kwargs": agg_kwargs},
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
