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
      - 'splitVFLc': trainable-module network (Dense+ReLU hidden layer,
        Dense+Sigmoid output), label party has no features of its own.
      - 'splitVFL': trainable-module network, label party also
        contributes its own features.

    For every architecture, non_secure mode is a direct plaintext
    replica of the same model secure mode trains under MPC - for
    aggVFLc/aggVFL that's the single fixed-aggregation linear score;
    for splitVFLc/splitVFL it's the one joint Dense+ReLU hidden layer
    over every party's concatenated features (each feature party holds
    its own row-slice of the shared weight matrix and exchanges a
    linear partial pre-activation - see train_splitvflc_vanilla.py /
    train_splitvfl_vanilla.py for the exact protocol). So comparing
    predictions/accuracy between secure and non_secure is a valid
    sanity check for any architecture.

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
        it reveals a per-hidden-unit partial pre-activation vector and
        gradient vector per epoch (a stronger signal than a single
        score, though still only a linear partial - not raw features).

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
    # F02: production fuzzy training is PAUSED. Diagnosed live
    # (2026-09-27): the fuzzy circuit can bind more than one of a
    # party's own local rows to the same alignment key, and different
    # parties' accepted-key SETS were not even consistent with each
    # other for one run of identical input (one party had an accepted
    # entity none of the others had any row bound to at all). A local,
    # per-party guard exists in the daemon (rejects training when a
    # party sees a duplicate on its own side) but is NOT sufficient by
    # itself: in this architecture only LP computes the shared training
    # mask, from its OWN local view - if FP1 or FP2 detects a problem
    # and blanks its own contribution while LP does not detect anything
    # on ITS side, training still runs, combining FP1/FP2's zeroed
    # features with LP's real labels under a mask that still marks
    # those positions "real" - silently invalid training, not a safely
    # skipped run. Re-enabling fuzzy training needs either a genuine
    # collective pre-training validity check (every party's local
    # validity AND their ordered alignment-key lists cross-verified,
    # with a shared - not per-party - decision to disable the whole
    # computation on any mismatch) or the corrected unique-triple
    # circuit, neither of which exists yet. Standalone PSI (psi_client_share/
    # psi_party_run, central()) is NOT affected - it's still the
    # diagnostic tool for validating alignment while this is worked on.
    if matching_method == "fuzzy":
        raise ValueError(
            "fuzzy-match training is temporarily disabled (F02: the fuzzy PSI "
            "circuit can produce duplicate/inconsistent alignment keys across "
            "parties, and no collective cross-party validity check exists yet "
            "to safely refuse training on that outcome) - use matching_method="
            "'exact' for training; fuzzy PSI itself (central()/psi_client_share) "
            "is unaffected and remains available for diagnosis"
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
    # n_samples ends up embedded directly in generated .mpc circuit
    # source text on the aggregator (see circuit_generator.py's
    # _validate_int, which is the actual enforcement point - this check
    # just fails fast here instead of only after dispatching to every
    # party). Vantage6 task kwargs are arbitrary JSON from whoever
    # submits the task, so nothing guarantees this arrived as a plain
    # int without this check.
    if n_samples is not None and (isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1):
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

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

    # R06 fix: ONE selection, reused for schema discovery AND for the
    # actual PSI+training dispatch below (both secure and non_secure) -
    # previously PSI always read "heart_vfl" while aggVFL/splitVFL
    # training separately, independently read "heart_vfl_aggvfl" for
    # FP2/LP, with nothing tying the two together - if the datasets
    # ever diverged, PSI could select a name missing from the training
    # file, or silently miss valid training rows present only there.
    # Keyed by CLIENT_ID (each daemon already self-selects its own role
    # this way, from its own PSI_CLIENT_ID env var - see F04's note
    # below) rather than by org_id, since client_kwargs is sent
    # identically to every client org and each daemon picks out its own
    # entry. The label party's own dataset only carries real feature
    # columns in the "heart_vfl_aggvfl" database (aggVFL/splitVFL); the
    # plain "heart_vfl" database gives it target+full_name only, which
    # report_schema correctly reports as 0 features.
    database_by_client_id = {
        0: "heart_vfl",
        1: "heart_vfl_aggvfl" if label_has_features else "heart_vfl",
        2: "heart_vfl_aggvfl" if label_has_features else "heart_vfl",
    }

    info(f"Central (train {architecture}, {privacy_mode}): starting training run "
         f"(run_id={run_id}, method={matching_method}, fuzzy_threshold={fuzzy_threshold}) - "
         f"features={feature_org_ids}, label={label_org_id}, aggregators={agg_org_ids}")

    # Schema discovery: before dispatching training, ask each party to
    # report its own row count and feature-column count (read straight
    # from its CSV).
    #
    # R11 fix: the ROW-COUNT half of this (and the party-role validation
    # it enables) now runs for BOTH privacy modes, not secure only - PSI
    # itself is the SAME private Rep3 circuit regardless of which
    # training math (secure or vanilla) consumes its output, so its own
    # row-count bound (max_entities) needs discovering identically either
    # way. Previously non_secure runs never got this at all: max_entities
    # was left unset in that branch, so vanilla training's PSI step
    # silently fell back to each daemon's own local default bound
    # (MAX_ENTITIES=320) no matter how large the real dataset was - a
    # raw dataset above that default would silently truncate PSI's own
    # candidate set with no error, before matching even ran.
    #
    # The FEATURE-COUNT half (n_feat_a/b/c, expected_n_features_by_client_id,
    # and the schema dict used to size/recompile the secure training
    # circuit) stays secure-only below - non_secure has no compiled
    # circuit to size and discovers its own columns locally at training
    # time.
    agg_kwargs = dict(kwargs)
    fp1_org, fp2_org = feature_org_ids[0], feature_org_ids[1]

    schema_tasks = {
        fp1_org: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {"database": database_by_client_id[0], "run_id": run_id}},
            organizations=[fp1_org], name=f"schema-{architecture}-{fp1_org}",
        )["id"],
        fp2_org: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {"database": database_by_client_id[1], "run_id": run_id}},
            organizations=[fp2_org], name=f"schema-{architecture}-{fp2_org}",
        )["id"],
        label_org_id: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {"database": database_by_client_id[2], "run_id": run_id}},
            organizations=[label_org_id], name=f"schema-{architecture}-{label_org_id}",
        )["id"],
    }
    schema_results = {org_id: client.wait_for_results(task_id=task_id)[0]
                       for org_id, task_id in schema_tasks.items()}
    # F04 / R11: feature_org_ids[0]/[1] and label_org_id only determine
    # which vantage6 ORGANIZATIONS receive the job dispatch - which
    # physical circuit slot (N_FEAT_A vs N_FEAT_B vs label) each one
    # plays is fixed independently by that host's own PSI_CLIENT_ID env
    # var, entirely outside this function's control. If a caller passes
    # org IDs in an order that doesn't match how this deployment's nodes
    # are actually configured, fp1_org's reported feature count would
    # silently get labeled n_feat_a and sent to the aggregator as the
    # expected size of whatever the ACTUAL client_id=0 party sends - a
    # receive-size mismatch at best, silently wrong-column training at
    # worst if the two parties' feature counts happen to coincide. This
    # validation matters identically for non_secure: database selection
    # (database_by_client_id) and feature ownership (which columns a
    # party's dataset holds) both depend on the SAME role assignment,
    # whichever privacy_mode is training. Each report_schema_run result
    # includes the reporting party's own client_id (see report_schema in
    # mpc_daemon_client_v2.py), so this is checked explicitly instead of
    # silently trusted.
    expected_client_id = {fp1_org: 0, fp2_org: 1, label_org_id: 2}
    for org_id, expected in expected_client_id.items():
        actual = schema_results[org_id].get("client_id")
        if actual != expected:
            raise ValueError(
                f"org {org_id} was expected to be client_id={expected} "
                f"(based on feature_org_ids/label_org_id order) but its "
                f"node reports client_id={actual!r} - feature_org_ids "
                f"must be [<the org whose node has PSI_CLIENT_ID=0>, "
                f"<PSI_CLIENT_ID=1>] and label_org_id must be the org "
                f"whose node has PSI_CLIENT_ID=2"
            )
    # PSI's own row-count bound (a public upper bound on any single
    # party's RAW candidate row count, before matching - separate from
    # n_samples below, which bounds the MATCHED intersection and can't be
    # safely auto-computed the same way). This one CAN be safely
    # auto-computed: raw row counts are already known from the schema
    # discovery just above, with no chicken-and-egg problem (unlike the
    # matched count, raw counts don't need PSI to run first). Rounded up
    # to the next multiple of 50 with at least 50 rows of headroom, so
    # the bound never lands exactly on any one party's true count.
    raw_row_counts = [schema_results[fp1_org]["n_rows"], schema_results[fp2_org]["n_rows"],
                       schema_results[label_org_id]["n_rows"]]
    max_entities = ((max(raw_row_counts) // 50) + 2) * 50
    agg_kwargs["max_entities"] = max_entities
    info(f"Central (train {architecture}, {privacy_mode}): discovered raw row counts "
         f"{raw_row_counts}, using PSI bound max_entities={max_entities}")

    if privacy_mode == "secure":
        n_feat_a = schema_results[fp1_org]["n_features"]
        n_feat_b = schema_results[fp2_org]["n_features"]
        n_feat_c = schema_results[label_org_id]["n_features"] if label_has_features else 0
        # R06 fix (residual gap): schema discovery is a separate, earlier
        # task from the training task that actually reads the CSV again -
        # nothing stops the file from changing in between (a boundary
        # reading-once-during-training alone can't close, since it's a
        # different task instance from discovery). Each party's OWN
        # expected feature count, so the training task can compare its
        # freshly-read data against what was discovered and fail clearly
        # on a mismatch instead of silently training on a shape schema
        # discovery never saw.
        expected_n_features_by_client_id = {0: n_feat_a, 1: n_feat_b, 2: n_feat_c}
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
        # here - omitting it keeps today's proven default shape. If the
        # true intersection ever exceeds this bound, run_train_client*
        # raises clearly rather than silently truncating (see
        # mpc_daemon_client_v2.py) - unchanged by this fix.
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
        # debug is threaded down to each client party's own subtask (not
        # just used below to filter central_train()'s own top-level
        # output) since that subtask's result is independently queryable
        # in vantage6 regardless of what central_train() itself returns -
        # without this, debug=False here would still leave raw
        # predictions sitting in every client subtask's own result.
        client_kwargs = dict(kwargs, algorithm=algorithm, max_entities=max_entities, debug=debug,
                              database_by_client_id=database_by_client_id,
                              expected_n_features_by_client_id=expected_n_features_by_client_id)
        if n_samples is not None:
            client_kwargs["n_samples_bound"] = n_samples
    else:
        # R06 fix: non_secure (vanilla) training needs the SAME database
        # selection as secure mode - previously this branch sent the raw
        # kwargs dict untouched, so every vanilla training path always
        # read "heart_vfl" regardless of architecture, hitting the exact
        # PSI/training divergence bug for aggVFL/splitVFL.
        # R11 fix: max_entities is now forwarded here too (see the
        # discovery block above, which computes it unconditionally) -
        # previously this branch never sent it, so vanilla training's PSI
        # step always used this daemon's own local default bound instead
        # of a bound sized for the real dataset. No
        # expected_n_features_by_client_id here: that field feeds the
        # secure-mode-only schema-discovery-staleness check; non_secure
        # has no compiled circuit for it to protect.
        client_kwargs = dict(kwargs, max_entities=max_entities, database_by_client_id=database_by_client_id)

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
                input_={"method": spec["worker_action"], "kwargs": client_kwargs},
                organizations=[org_id],
                name=f"train-{architecture}-worker-{org_id}",
            )
            tasks[org_id] = t["id"]
        t = client.task.create(
            input_={"method": spec["coordinator_action"], "kwargs": client_kwargs},
            organizations=[label_org_id],
            name=f"train-{architecture}-coordinator-{label_org_id}",
        )
        tasks[label_org_id] = t["id"]
        # Row alignment still needs the real Rep3 PSI circuit even in
        # non_secure mode - only the training math itself skips MPC.
        # R11 fix: dispatch with agg_kwargs (carries max_entities), not
        # the bare kwargs - previously the aggregator's own PSI party
        # never received the discovered row-count bound either, so it
        # fell back to its own local default regardless of the real
        # dataset size.
        for org_id in agg_org_ids:
            t = client.task.create(
                input_={"method": "psi_party_run", "kwargs": agg_kwargs},
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
    # secure mode: client subtasks now report a predictions_hash instead
    # of raw predictions unless debug=True (see _redact_training_result
    # in mpc_daemon_client_v2.py) - Rep3's correctness cross-check only
    # needs to confirm every client's predictions came out byte-identical,
    # not what they actually are, so comparing hashes works the same
    # whether or not debug also requested the raw values. non_secure
    # (vanilla) mode's leak is deliberately left as-is for now (already
    # documented as insecure-by-design), so its subtasks still return raw
    # predictions with no hash - compared directly as before.
    #
    # R05 fix: predictions_agree (like central.py's clients_agree) is
    # only computed over parties that actually completed - a prior
    # version dropped failed/missing clients from both the agreement
    # check and the party count, so e.g. 1 success + 2 failures could
    # still report predictions_agree=True by trivially comparing one
    # value to itself. clients_ok now requires EVERY expected client to
    # have completed, and predictions_agree is only ever True when
    # clients_ok is also True.
    client_statuses = {org_id: (results.get(org_id) or {}).get("status") for org_id in client_org_ids}
    aggregators_ok = all(
        results.get(org_id) and results.get(org_id).get("status") == "complete"
        for org_id in agg_org_ids
    )

    # EMPTY-INTERSECTION fix: an empty accepted intersection (no shared entities, or -
    # for fuzzy matching - every candidate excluded as ambiguous) is an
    # ordinary, valid outcome, not a failure. Each client party
    # independently derives its own aligned count from its own PSI result
    # and reports status="no_matches" when that count is zero (see
    # run_train_client / run_vanilla_train's matching fix) - checked here
    # as agreement across ALL clients, never inferred from just one
    # client's result. A party that genuinely failed reports some OTHER
    # status ("error", or missing entirely), which prevents this branch
    # from firing at all - that case falls through to the ordinary
    # success/incomplete logic below and is reported as a failure, not
    # silently reinterpreted as an empty intersection. aggregators_ok is
    # required too: their (fixed-shape, privacy-preserving) training
    # circuit still runs to completion even when the real intersection is
    # empty - see mpc_daemon_client_v2.py's matching fix for why the
    # client side still completes that same connection rather than
    # skipping it - so a genuinely empty run still expects them to report
    # "complete" like any other run; an aggregator failure alongside
    # all-clients-no_matches is still a real problem worth reporting as
    # incomplete, not masked by the empty-intersection outcome.
    #
    # Known open limitation: this check only DETECTS disagreement among
    # clients after their tasks have already run and returned - it cannot
    # PREVENT one party's vanilla-mode task from hanging if it and
    # another party land on different aligned counts for the same run
    # (exact match's counts are consistent by construction; the current
    # production fuzzy circuit's ambiguity-exclusion rule has not been
    # independently verified to guarantee that same consistency across
    # all 3 parties in every case - see mpc_daemon_client_v2.py's
    # run_vanilla_train comment). Closing that gap needs a real
    # pre-training coordination signal between parties, not just
    # after-the-fact result comparison; that is not implemented here.
    #
    # Message wording: NOT "training skipped" - in secure mode the padded
    # computation genuinely executes (aggregators complete their circuit
    # normally, on an all-zero mask); nothing was skipped there. What's
    # true in BOTH modes is that no usable model came out of it.
    all_no_matches = all(s == "no_matches" for s in client_statuses.values())
    if all_no_matches and aggregators_ok:
        message = "No matching records; no model result produced."
        info(f"Central (train {architecture}, {privacy_mode}): {message}")
        return {
            "summary": [
                {"metric": "run_id", "value": run_id},
                {"metric": "overall_status", "value": "no_matches"},
                {"metric": "architecture", "value": architecture},
                {"metric": "privacy_mode", "value": privacy_mode},
                {"metric": "matching_method", "value": matching_method},
                {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
                {"metric": "aligned_count", "value": 0},
                {"metric": "message", "value": message},
            ],
            "run_id": run_id,
            "overall_status": "no_matches",
            "architecture": architecture,
            "privacy_mode": privacy_mode,
            "matching_method": matching_method,
            "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
            "aligned_count": 0,
            "predictions_agree": None,
            "clients_ok": True,
            "aggregators_ok": True,
            "party_errors": {},
            "message": message,
        }

    clients_ok = all(s == "complete" for s in client_statuses.values())

    predictions_by_org = {}
    for org_id in client_org_ids:
        r = results.get(org_id)
        if client_statuses[org_id] == "complete":
            predictions_by_org[org_id] = (
                r.get("predictions_hash") if privacy_mode == "secure" else r.get("predictions")
            )
            if aligned_count is None:
                aligned_count = r.get("aligned_count")

    def _agree_key(p):
        return p if privacy_mode == "secure" else tuple(p)

    predictions_agree = clients_ok and len(
        {_agree_key(p) for p in predictions_by_org.values() if p is not None}
    ) == 1

    # aggregators_ok computed earlier (needed there for the no_matches check too).
    overall_status = "success" if (clients_ok and aggregators_ok and predictions_agree) else "incomplete"

    # Sanitized per-party errors, same rationale as central.py's R05
    # fix: status + message only, surfaced for every party that did NOT
    # complete regardless of debug.
    party_errors = {}
    for org_id in list(client_org_ids) + list(agg_org_ids):
        r = results.get(org_id)
        status = (r or {}).get("status")
        if status != "complete":
            party_errors[org_id] = {
                "status": status,
                "message": (r or {}).get("message", "no result received"),
            }

    info(f"Central (train {architecture}, {privacy_mode}): training {overall_status} - "
         f"aligned_count={aligned_count} (predictions_agree={predictions_agree}, "
         f"clients_ok={clients_ok}, aggregators_ok={aggregators_ok})"
         + (f", errors={party_errors}" if party_errors else ""))

    output = {
        "summary": [
            {"metric": "run_id", "value": run_id},
            {"metric": "overall_status", "value": overall_status},
            {"metric": "architecture", "value": architecture},
            {"metric": "privacy_mode", "value": privacy_mode},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "aligned_count", "value": aligned_count},
            {"metric": "predictions_agree", "value": predictions_agree},
            {"metric": "clients_ok", "value": clients_ok},
            {"metric": "aggregators_ok", "value": aggregators_ok},
        ],
        "run_id": run_id,
        "overall_status": overall_status,
        "architecture": architecture,
        "privacy_mode": privacy_mode,
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
        "aligned_count": aligned_count,
        "predictions_agree": predictions_agree,
        "clients_ok": clients_ok,
        "aggregators_ok": aggregators_ok,
        "party_errors": party_errors,
    }

    if debug:
        # secure mode: raw predictions are only present in a client
        # subtask's own result when debug=True made it all the way down
        # to that party (see client_kwargs above) - non_secure mode
        # never redacted them in the first place.
        output["predictions"] = next(
            (results[org_id].get("predictions") for org_id in client_org_ids
             if results.get(org_id) and results[org_id].get("predictions") is not None),
            None,
        )
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
