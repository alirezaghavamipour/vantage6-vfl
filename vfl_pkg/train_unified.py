import json
import re
import uuid

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info

from . import _psi_capacity
from . import datasets


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)

# EXPERIMENTAL fuzzy PSI (F02/F03 redesign): strict unique-triple
# acceptance + exhaustive matching + R13 consecutive alignment IDs,
# reached only via matching_method="fuzzy_experimental" - completely
# separate from matching_method="fuzzy" (the OLD circuit, still paused
# below).
#
# Manual capacity selection (Phase 1, 2026-09-28): 100/350/700/1000,
# each independently compiled/validated - see _psi_capacity.py for the
# shared capacity set and daemons/psi_fuzzy_capacity.py for the
# matching daemon-side program/port mapping. psi_capacity below lets a
# caller (the UI, or a direct script) select one explicitly; None uses
# _psi_capacity.DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY (350), preserving
# this deployment's exact pre-selection behavior.

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


def _collective_fuzzy_experimental_check(client, client_org_ids, agg_org_ids, run_id, max_entities,
                                          database_by_client_id, expected_client_id, fuzzy_threshold,
                                          capacity_mode="manual"):
    """F02 fix (fuzzy_experimental only): a genuine COLLECTIVE pre-training
    validity check, run to completion BEFORE any training task is
    dispatched to anyone. Per the review that paused production fuzzy
    training: a per-party local guard alone is not a safe abort mechanism
    (in this architecture only LP computes the shared training mask, from
    its OWN local view - another party privately blanking its own
    contribution does not stop LP from still running training with a mask
    that marks those positions "real"). This instead runs
    matching_method="fuzzy_experimental" PSI standalone first (via
    psi_client_share, reveal_align_keys=True so it can see the non-PII
    matched_align_keys/local_valid fields, never matched_ids), and only
    allows training to be dispatched at all when EVERY party is
    individually valid (no local duplicate alignment key) AND all three
    parties' accepted alignment-ID SETS agree exactly - not just their
    counts, which "Equal counts are insufficient" for. The
    psi_fuzzy_unique_triple circuit's own R13 consecutive-numbering
    scheme (0..K-1, the same for every party under a consistent match)
    makes that set comparison direct, with no PII involved.

    Any single failure means NO training task is dispatched to ANY party -
    one shared, coordinated reject, not a per-party decision made
    independently by each computing party's own mask.

    G01 fix: also collects each party's mapping_digest (an opaque hash
    of its own accepted (local_index, alignment_id) pairing - see
    mpc_daemon_client_v2.py's _mapping_digest) and, only once every
    other check here has passed, returns it keyed by CLIENT_ID as
    approved_mapping_digest_by_client_id. central_train forwards this
    to training UNCHANGED - it is the value _load_alignment_artifact
    later requires an exact match against, recomputed from whatever is
    actually on disk at consumption time. This is what makes the digest
    a genuine approval binding rather than something read back from the
    same (mutable) artifact file it's meant to protect: it never
    round-trips through that file at all.

    Returns (ok: bool, detail: dict, approved_mapping_digest_by_client_id:
    dict) - detail holds only non-PII per-party status/local_valid/
    intersection_size/id-agreement info, safe to return directly in
    central_train's own result. approved_mapping_digest_by_client_id is
    None when ok is False (nothing to approve).

    k=1 Stage 3 fix: fuzzy_threshold is now a required parameter,
    threaded into both dispatched jobs' kwargs below. Before this fix,
    this function's own psi_client_share/psi_party_run dispatch never
    included fuzzy_threshold at all, silently falling back to that
    wrapped function's own default (2) regardless of what threshold the
    caller actually requested - so this collective check (and the
    artifact it approves and writes) always ran at k=2 even when
    central_train() was called with fuzzy_threshold=1, while the
    SEPARATE training dispatch just below correctly forwarded the real
    requested threshold. The result was every k=1 secure training call
    failing collectively at the artifact-load step with "alignment
    artifact was computed for fuzzy_threshold=2, but training requested
    1" - a real, previously-undetected gap in threshold propagation,
    caught by Stage 3's own real end-to-end central_train() test (Stage
    1/2 never exercised this function directly). Fixed by requiring the
    caller to pass fuzzy_threshold explicitly, same no-default
    convention _load_alignment_artifact's own approved_mapping_digest
    parameter already uses, for the same reason: silently defaulting
    here would defeat the point as surely as silently skipping would."""
    # The computing parties must ALSO be dispatched (psi_party_run) -
    # without this the psi_fuzzy_unique_triple circuit is never launched
    # on the aggregators at all, and every client's psi_client_share
    # fails immediately with a connection-refused error (a real bug
    # caught live 2026-09-27: this function originally only dispatched
    # to client_org_ids, mirroring central.py's OWN two-loop pattern was
    # missed for this new function).
    # database_by_client_id: F02 fix - this PSI run must read the SAME
    # dataset a later training call for this party will (R06's own
    # rationale), or an artifact this check "approves" here (see
    # _write_alignment_artifact in mpc_daemon_client_v2.py) could be
    # computed against the wrong data for architectures where the label
    # party's database differs (aggVFL/splitVFL) - without this, training
    # would correctly detect the mismatch via its own database check and
    # refuse to proceed, but the run would fail confusingly rather than
    # validating the right thing the first time.
    tasks = {
        org_id: client.task.create(
            input_={"method": "psi_client_share", "kwargs": {
                "matching_method": "fuzzy_experimental",
                "fuzzy_threshold": fuzzy_threshold,
                "run_id": run_id,
                "max_entities": max_entities,
                "reveal_align_keys": True,
                "database_by_client_id": database_by_client_id,
                "capacity_mode": capacity_mode,
            }},
            organizations=[org_id], name=f"fuzzy-experimental-prealign-{org_id}",
        )["id"]
        for org_id in client_org_ids
    }
    agg_tasks = {
        org_id: client.task.create(
            input_={"method": "psi_party_run", "kwargs": {
                "matching_method": "fuzzy_experimental",
                "fuzzy_threshold": fuzzy_threshold,
                "run_id": run_id,
                "max_entities": max_entities,
                "capacity_mode": capacity_mode,
            }},
            organizations=[org_id], name=f"fuzzy-experimental-prealign-agg-{org_id}",
        )["id"]
        for org_id in agg_org_ids
    }
    results = {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        results[org_id] = res[0] if res else None
    agg_results = {}
    for org_id, task_id in agg_tasks.items():
        res = client.wait_for_results(task_id=task_id)
        agg_results[org_id] = res[0] if res else None

    aggregators_ok = all(
        agg_results.get(org_id) and agg_results.get(org_id).get("status") == "complete"
        for org_id in agg_org_ids
    )

    per_party = {}
    for org_id in client_org_ids:
        r = results.get(org_id) or {}
        per_party[org_id] = {
            "status": r.get("status"),
            "local_valid": r.get("local_valid"),
            "intersection_size": r.get("intersection_size"),
        }

    all_complete = aggregators_ok and all(p["status"] == "complete" for p in per_party.values())
    all_locally_valid = all_complete and all(p["local_valid"] is True for p in per_party.values())
    sizes = {p["intersection_size"] for p in per_party.values()} if all_complete else set()
    counts_agree = len(sizes) == 1
    k = next(iter(sizes)) if counts_agree else None

    ids_agree = False
    if all_locally_valid and counts_agree:
        expected = set(range(k))
        for org_id in client_org_ids:
            id_set = set((results.get(org_id) or {}).get("matched_align_keys") or [])
            per_party[org_id]["id_set_matches_expected"] = (id_set == expected)
        ids_agree = all(p["id_set_matches_expected"] for p in per_party.values())

    ok = all_complete and all_locally_valid and counts_agree and ids_agree
    detail = {
        "per_party": per_party,
        "aggregators_ok": aggregators_ok,
        "all_complete": all_complete,
        "all_locally_valid": all_locally_valid,
        "counts_agree": counts_agree,
        "ids_agree": ids_agree,
        "aligned_count": k if ok else None,
    }
    # G01 fix: only ever built when the check as a whole passed - an
    # approved digest for a run that DIDN'T pass would be meaningless
    # (and training is never dispatched for it anyway).
    approved_mapping_digest_by_client_id = None
    if ok:
        approved_mapping_digest_by_client_id = {
            str(expected_client_id[org_id]): (results.get(org_id) or {}).get("mapping_digest")
            for org_id in client_org_ids
        }
    return ok, detail, approved_mapping_digest_by_client_id


def _describe_model(descriptor):
    """Plain description of the delivered model from the public (structural)
    descriptor in the share headers: no names, scales or values."""
    model = descriptor.get("model") or {}
    slots = {s.get("client_id"): s.get("count") for s in descriptor.get("input_slots") or []}
    n_inputs = sum(c for c in slots.values() if isinstance(c, int))
    inputs = f"{n_inputs} inputs (FP1 {slots.get(0)}, FP2 {slots.get(1)}, LP {slots.get(2)})"
    output = "probability (sigmoid)" if model.get("output") == "sigmoid" else "value (linear)"
    sizes = []
    for entry in descriptor.get("layout") or []:
        size = 1
        for dim in entry.get("shape") or []:
            size *= dim
        sizes.append((entry.get("name"), size))
    n_params = sum(s for _, s in sizes)
    if model.get("family") == "one_layer":
        kind = "logistic regression" if model.get("output") == "sigmoid" else "linear regression"
        return f"{kind}: {inputs} -> 1 {output}", f"{n_params} values ({n_params - 1} weights + 1 bias)"
    if model.get("family") == "two_layer":
        hidden = model.get("hidden")
        weights = sum(s for name, s in sizes if name.endswith(".W"))
        return (f"two-layer neural network: {inputs} -> {hidden} hidden units (ReLU) -> 1 {output}",
                f"{n_params} values ({weights} weights + {n_params - weights} biases)")
    return "unknown model structure", f"{n_params} values"


def _secure_summary(run_id, overall_status, architecture, dataset, matching_method, fuzzy_threshold, aligned_count,
                    clients_ok, aggregators_ok, researcher_output, shares, preprocessing):
    """The secure-mode summary table: what was trained and which encrypted
    packages arrived, from the envelopes' public headers only. This function
    cannot open the packages, so it never claims the model is valid or that
    the signatures verify - that happens on the researcher's machine."""
    delivered = (researcher_output or {}).get("status") == "encrypted_shares_delivered"
    share_heads = [e.get("header") or {} for e in shares]
    prep_heads = [e.get("header") or {} for e in preprocessing]
    rows = [
        ("run_id", run_id),
        ("delivery", "Training finished; all encrypted packages received" if delivered
         else "No model delivered - see party_errors (nothing usable was released)"),
        ("overall_status", overall_status),
        ("architecture", architecture),
        ("dataset", dataset),
        ("privacy_mode", "secure"),
        ("matching_method", matching_method),
    ]
    if matching_method in ("fuzzy", "fuzzy_experimental"):
        rows.append(("fuzzy_threshold", fuzzy_threshold))
    rows += [("aligned_count", aligned_count), ("clients_ok", clients_ok), ("aggregators_ok", aggregators_ok)]
    roles = {0: "FP1", 1: "FP2", 2: "LP"}
    rows.append(("computing_party_packages", f"{len(share_heads)} of 3 received" + (
        f" (parties {', '.join(str(i) for i in sorted(h.get('share_index') for h in share_heads))})"
        if share_heads else "")))
    rows.append(("data_holder_packages", f"{len(prep_heads)} of 3 received" + (
        f" ({', '.join(roles.get(h.get('client_id'), '?') for h in sorted(prep_heads, key=lambda h: str(h.get('client_id'))))})"
        if prep_heads else "")))
    if delivered and share_heads:
        descriptor = share_heads[0].get("descriptor") or {}
        model, params = _describe_model(descriptor)
        rows += [("model", model), ("parameters", params)]
        recipients = {json.dumps(h.get("recipient"), sort_keys=True) for h in share_heads + prep_heads}
        recipient = (share_heads[0].get("recipient") or {}) if len(recipients) == 1 else None
        rows.append(("encrypted_for", f"{recipient.get('organization')}, key {str(recipient.get('key_sha256'))[:16]}…"
                     if recipient else "packages name different recipients - do not use"))
        builds = {json.dumps(h.get("descriptor", {}).get("build"), sort_keys=True) for h in share_heads}
        build = descriptor.get("build") or {}
        rows.append(("circuit_build", f"generator v{build.get('generator_version')}, bytecode "
                     f"{str(build.get('bytecode_sha256'))[:16]}… ("
                     + ("same build reported by all 3 computing parties" if len(builds) == 1
                        else "computing parties report different builds") + ")"))
        rows.append(("signatures", "attached to every package; not checked by the server - checked against your "
                                   "pinned certificates when you open the model"))
        rows.append(("next_step", f"open on your own machine: vfl-open {run_id}"))
    return [{"metric": k, "value": v} for k, v in rows]


def _researcher_output_preflight(client, org_ids, researcher_key_sha256, run_id):
    """Dispatches researcher_output_check to every party at once; returns
    {org_id: {"status", "reason"}} for each party that is not ready (empty
    when all are)."""
    tasks = {org_id: client.task.create(
        input_={"method": "researcher_output_check",
                "kwargs": {"recipient_key_sha256": researcher_key_sha256, "run_id": run_id}},
        organizations=[org_id], name=f"researcher-output-check-{org_id}")["id"] for org_id in org_ids}
    not_ready = {}
    for org_id, task_id in tasks.items():
        res = client.wait_for_results(task_id=task_id)
        r = res[0] if res else None
        if not (isinstance(r, dict) and r.get("status") == "complete" and r.get("researcher_output") == "ready"):
            reason = (r or {}).get("reason")
            not_ready[org_id] = {"status": (r or {}).get("status", "no result"),
                                 "reason": reason if isinstance(reason, str) and re.fullmatch(r"[A-Za-z]+", reason)
                                 else None}
    return not_ready


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
    psi_capacity: int = None,
    psi_capacity_mode: str = "manual",
    n_samples: int = None,
    algorithm: str = "logistic",
    debug: bool = False,
    dataset: str = datasets.DEFAULT_DATASET,
    researcher_key_sha256: str = None,
):
    """
    Train a vertical federated learning model - pick which of the 4
    architectures and whether to run it privately (Rep3 MPC) or as a
    deliberately insecure plaintext baseline for comparison.

    dataset: a benchmark id ('bcw_exact' by default, 'diabetes_fuzzyk2',
    ...). Every party reads '<id>_labelonly' for aggVFLc/splitVFLc or
    '<id>_distributed' for aggVFL/splitVFL, for schema discovery,
    snapshots, PSI and training alike. The selection is checked
    against algorithm and PSI capacity before any task is dispatched, and
    in secure mode n_samples defaults to the benchmark's rows per party.

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

    psi_capacity_mode (matching_method="fuzzy_experimental" only):
    "manual" (default) or "automatic". Kept as a SEPARATE argument from
    psi_capacity below - rather than letting psi_capacity itself accept
    a string sentinel like "auto" - specifically so the UI's existing
    integer capacity dropdown keeps one consistent type; overloading it
    with an occasional string value is a schema-mismatch risk for the
    UI layer, not just a Python typing nicety.

    psi_capacity (matching_method="fuzzy_experimental",
    psi_capacity_mode="manual" only): which precompiled PSI capacity to
    use - 100, 350, 700, or 1000 raw candidate rows per party. None
    (always the case from the UI) uses the selected dataset's required
    capacity - bcw 100, diabetes 350, credit 1000; an explicit value is
    for direct calls only. Schema discovery (report_schema_run) is
    dispatched to every client first, to learn the real raw row counts
    in the clear; rejection then happens clearly, before any PSI or
    training execution task is launched, if those counts exceed the
    selected capacity - never silently truncated to fit. Ignored when
    psi_capacity_mode="automatic" (see below) - manual mode's own
    precompiled-circuit behavior is otherwise completely unchanged.

    When psi_capacity_mode="automatic" (Phase 2 Stage 3/4), the exact
    capacity is negotiated privately instead: schema discovery never
    learns row counts in this mode (only feature/column counts) - the
    maximum row count across all 3 parties IS revealed (that is the
    mechanism), but no individual party's count ever is. That maximum
    becomes max_entities directly (any integer 1..1000). The
    corresponding circuit is then compiled on demand - first use at a
    given exact capacity takes noticeably longer (compilation time),
    later runs at the same exact capacity reuse the cached build - and
    every aggregator must independently report an identical build
    fingerprint and bytecode hash (the build-agreement barrier) before
    PSI is dispatched - a missing/failed/disagreeing build aborts the
    whole run, the same coordinated-reject shape as a failed collective
    alignment check below. Initial support: matching_method=
    "fuzzy_experimental", fuzzy_threshold=2 only, up to 1000 raw rows
    per party.

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

    algorithm (both privacy modes):
      - 'logistic' (default): binary classification - the label must be
        an exact 0/1 value, predictions are 0/1 classifications.
      - 'linear': regression - the label is a continuous value,
        normalized and fixed-point encoded the same way every feature
        column already is.

    Output in secure mode (researcher-only, Stage 2): no prediction or
    model value is revealed to any data holder or computing party. The
    data holders learn only the collective validity bit. The trained
    weights and biases leave the computation only as three output shares,
    one per computing party, each encrypted to the researcher's key and
    signed by that party; each data holder adds an envelope, encrypted
    to the same key and signed by that data holder, with its column
    names, normalization scales and (label party) the label encoding.
    researcher_key_sha256 (required in secure mode) is the SHA-256
    fingerprint of the researcher's public key; every host compares it
    with the key configured on that host and refuses on a mismatch.

    This function's normal result (no debug needed) forwards those six
    envelopes under "researcher_output". Its status says only that the
    encrypted shares were delivered ("encrypted_shares_delivered"):
    neither this function nor the server can see, and so cannot assert,
    whether they reconstruct into a valid model. That is decided by the
    researcher's own tool (daemons/researcher_reconstruct.py), which
    also refuses an invalid run. A collectively rejected run (invalid
    alignment) yields no researcher output at all.

    non_secure mode is unchanged: predictions are revealed to every
    client party and compared here, and debug=True adds the per-party
    results.
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
    if matching_method == "fuzzy_experimental":
        # k=1 addition: previously unvalidated at this layer - see
        # _psi_capacity.SUPPORTED_FUZZY_EXPERIMENTAL_THRESHOLDS's own
        # docstring for why this is a separate constant/check from the
        # "fuzzy" branch just above.
        _psi_capacity.validate_fuzzy_experimental_threshold(fuzzy_threshold)
    if psi_capacity_mode not in ("manual", "automatic"):
        raise ValueError(
            f"psi_capacity_mode={psi_capacity_mode!r} is not supported "
            f"(choose 'manual' or 'automatic')"
        )
    if psi_capacity_mode == "automatic":
        # Normalizes onto the SAME internal sentinel Stage 3's own
        # direct-call interface already uses (psi_capacity="auto") -
        # is_automatic_capacity below is unchanged either way, so a
        # direct caller passing psi_capacity="auto" without setting
        # this new argument keeps working exactly as already tested.
        psi_capacity = _psi_capacity.AUTOMATIC_CAPACITY_SENTINEL
    psi_capacity = datasets.resolve_psi_capacity(dataset, matching_method, psi_capacity)
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
    if privacy_mode == "secure" and not (isinstance(researcher_key_sha256, str)
                                         and re.fullmatch(r"[0-9a-f]{64}", researcher_key_sha256)):
        raise ValueError(
            "secure training releases the model only to the researcher: researcher_key_sha256 must be "
            "the SHA-256 fingerprint (64 lowercase hex characters) of the researcher's public key"
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
    n_samples = datasets.validate_selection(dataset, algorithm, matching_method, psi_capacity,
                                            privacy_mode=privacy_mode, n_samples=n_samples)
    # The label party may also be ticked as a feature party (it does hold
    # features in aggVFL/splitVFL); its features are used through its role
    # as label party, so it is dropped here, as are duplicates.
    feature_org_ids = [org for org in dict.fromkeys(feature_org_ids) if org != label_org_id]
    if len(feature_org_ids) != 2:
        raise ValueError(
            "Feature parties must be the two organizations other than the label party "
            f"(e.g. FP1 and FP2); got {feature_org_ids} besides label party {label_org_id}"
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
    database_by_client_id = datasets.database_by_client_id(dataset, label_has_features)

    info(f"Central (train {architecture}, {privacy_mode}): starting training run "
         f"(run_id={run_id}, dataset={dataset}, databases={database_by_client_id}, "
         f"method={matching_method}, fuzzy_threshold={fuzzy_threshold}, n_samples={n_samples}) - "
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
    if privacy_mode == "secure":
        # Stage 2 researcher-only output: every data holder and computing
        # party must be configured for this researcher before any PSI or
        # training runs; otherwise the run is refused here, with nothing
        # computed (a host that is not ready would otherwise refuse only
        # mid-run and leave the others waiting on it).
        not_ready = _researcher_output_preflight(client, client_org_ids + list(agg_org_ids),
                                                 researcher_key_sha256, run_id)
        if not_ready:
            message = "researcher output is not configured for this key on every party; nothing was computed"
            info(f"Central (train {architecture}, secure): {message} - {not_ready}")
            return {
                "summary": [
                    {"metric": "run_id", "value": run_id},
                    {"metric": "overall_status", "value": "researcher_output_rejected"},
                    {"metric": "architecture", "value": architecture},
                    {"metric": "privacy_mode", "value": privacy_mode},
                    {"metric": "message", "value": message},
                ],
                "run_id": run_id,
                "overall_status": "researcher_output_rejected",
                "architecture": architecture,
                "dataset": dataset,
                "privacy_mode": privacy_mode,
                "party_errors": not_ready,
                "researcher_output": {"status": "not_delivered", "run_id": run_id},
                "message": message,
            }

    agg_kwargs = dict(kwargs)
    fp1_org, fp2_org = feature_org_ids[0], feature_org_ids[1]

    # Phase 2 Stage 3/4: automatic capacity mode - psi_capacity == "auto"
    # negotiates max_entities privately (see
    # _psi_capacity.negotiate_automatic_capacity, shared with central())
    # instead of computing it from raw row counts, so schema discovery
    # must never expose n_rows in this mode. Checked once here, reused
    # for both the schema-dispatch kwargs below and the branch further
    # down that decides how max_entities gets resolved.
    is_automatic_capacity = (
        matching_method == "fuzzy_experimental"
        and psi_capacity == _psi_capacity.AUTOMATIC_CAPACITY_SENTINEL
    )

    # Bug #3 fix: automatic mode's capacity negotiation (which retains
    # each party's dataset snapshot as a side effect - see
    # retain_dataset_snapshot in mpc_daemon_client_v2.py) now runs BEFORE
    # schema discovery, so schema discovery (dispatched just below) can
    # read from that SAME already-retained snapshot instead of its own
    # separate, potentially-diverged fresh CSV read. Manual mode is
    # unaffected - schema discovery still runs first there (negotiation
    # is skipped entirely for manual mode), exactly as before. One real
    # tradeoff: the org-id/client_id role-mismatch check just below
    # (which depends on schema_results) now runs AFTER a full automatic-
    # mode negotiation instead of before it - a role-configuration error
    # is still caught and rejected correctly, just one step later than
    # in manual mode, since there's no cheaper way to learn client_id
    # before schema discovery itself runs.
    if is_automatic_capacity:
        # Row counts were never learned in the clear (include_row_count
        # will be False below) - max_entities instead comes from private
        # row-count discovery + the build-agreement barrier. Any
        # failure here is a coordinated reject: no PSI or training task
        # is dispatched to any party, exactly like a failed
        # _collective_fuzzy_experimental_check below.
        negotiation_ok, max_entities, negotiation_detail = _psi_capacity.negotiate_automatic_capacity(
            client, client_org_ids, agg_org_ids, run_id, fuzzy_threshold,
            database_by_client_id=database_by_client_id,
        )
        info(f"Central (train {architecture}, {privacy_mode}): automatic capacity negotiation "
             f"{'PASSED' if negotiation_ok else 'FAILED'} - {negotiation_detail}")
        if not negotiation_ok:
            message = (
                "Automatic capacity negotiation failed - no PSI or training task was "
                "dispatched to any party, and no model result was produced. See "
                "'negotiation_detail' for the non-PII diagnostic (no individual row "
                "counts are ever included)."
            )
            return {
                "summary": [
                    {"metric": "run_id", "value": run_id},
                    {"metric": "overall_status", "value": "capacity_negotiation_error"},
                    {"metric": "architecture", "value": architecture},
                    {"metric": "dataset", "value": dataset},
                    {"metric": "privacy_mode", "value": privacy_mode},
                    {"metric": "matching_method", "value": matching_method},
                    {"metric": "message", "value": message},
                ],
                "run_id": run_id,
                "overall_status": "capacity_negotiation_error",
                "architecture": architecture,
                "dataset": dataset,
                "privacy_mode": privacy_mode,
                "matching_method": matching_method,
                "fuzzy_threshold": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None,
                "aligned_count": None,
                "message": message,
                "negotiation_detail": negotiation_detail,
            }
        raw_row_counts = None  # never learned in this mode - see include_row_count below
        info(f"Central (train {architecture}, {privacy_mode}): automatic capacity negotiated "
             f"max_entities={max_entities}")

    # capacity_mode="dynamic" (bug #3 fix, automatic mode only): reads
    # from the snapshot negotiation just retained above, instead of a
    # fresh CSV read.
    _schema_capacity_kwargs = {"capacity_mode": "dynamic"} if is_automatic_capacity else {}
    schema_tasks = {
        fp1_org: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {
                "database": database_by_client_id[0], "run_id": run_id,
                "include_row_count": not is_automatic_capacity, **_schema_capacity_kwargs,
            }},
            organizations=[fp1_org], name=f"schema-{architecture}-{fp1_org}",
        )["id"],
        fp2_org: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {
                "database": database_by_client_id[1], "run_id": run_id,
                "include_row_count": not is_automatic_capacity, **_schema_capacity_kwargs,
            }},
            organizations=[fp2_org], name=f"schema-{architecture}-{fp2_org}",
        )["id"],
        label_org_id: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {
                "database": database_by_client_id[2], "run_id": run_id,
                "include_row_count": not is_automatic_capacity, **_schema_capacity_kwargs,
            }},
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
    #
    # Bug #3 fix: automatic mode's own max_entities was already resolved
    # (and logged) by the negotiation block above, which now runs BEFORE
    # schema discovery - only the manual-mode resolution happens here.
    if not is_automatic_capacity:
        raw_row_counts = [schema_results[fp1_org]["n_rows"], schema_results[fp2_org]["n_rows"],
                           schema_results[label_org_id]["n_rows"]]
        max_entities = ((max(raw_row_counts) // 50) + 2) * 50
        if matching_method == "fuzzy_experimental":
            # Manual capacity selection (Phase 1, 2026-09-28): resolves
            # psi_capacity (explicit selection, or the default) against the
            # supported/validated set, and rejects clearly - before any
            # task is dispatched - if the real dataset's raw row count
            # exceeds it. ensure_psi_compiled on the aggregator side would
            # also reject a mismatched bound, but failing here is clearer
            # and avoids dispatching any task first. See _psi_capacity.py.
            #
            # Bug #2 fix: see central.py's matching comment - the
            # precompiled manual menu needs no headroom padding above the
            # actual max raw row count; passing the padded max_entities
            # here instead rejected datasets that actually fit (e.g. 50
            # rows needing max_entities=150, rejected at capacity=100).
            max_entities = _psi_capacity.resolve_capacity(psi_capacity, max(raw_row_counts), raw_row_counts)
        info(f"Central (train {architecture}, {privacy_mode}): discovered raw row counts "
             f"{raw_row_counts}, using PSI bound max_entities={max_entities}")
    agg_kwargs["max_entities"] = max_entities

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
        #
        # count-disclosure fix: capacity_mode is forwarded to the training
        # dispatch itself (not just _collective_fuzzy_experimental_check's
        # own prealignment PSI call below) - training never needed this
        # before (it reads the artifact _collective_fuzzy_experimental_check
        # already wrote rather than re-running PSI, so no port/program
        # selection depends on it), but the daemon now uses it to decide
        # whether an alignment-artifact validation failure's error message
        # may include this party's raw local_count - see
        # mpc_daemon_client_v2.py's _validate_artifact_schema(redact_counts=).
        # Without this, automatic mode's training-path errors would keep
        # disclosing local_count even though the PSI-path ones no longer do.
        client_kwargs = dict(kwargs, algorithm=algorithm, max_entities=max_entities, debug=debug,
                              database_by_client_id=database_by_client_id,
                              expected_n_features_by_client_id=expected_n_features_by_client_id,
                              capacity_mode="dynamic" if is_automatic_capacity else "manual",
                              recipient_key_sha256=researcher_key_sha256)
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
        # algorithm='linear' now implemented for non_secure too (was
        # previously rejected before reaching here) - forwarded the same
        # way the secure branch above already does, so the vanilla
        # daemon functions know which label encoding/activation to use.
        client_kwargs = dict(kwargs, algorithm=algorithm, max_entities=max_entities, database_by_client_id=database_by_client_id,
                              capacity_mode="dynamic" if is_automatic_capacity else "manual")

    # F02 fix (fuzzy_experimental only): collective pre-training validity
    # check, run to completion before ANY training or PSI-computing-party
    # task is dispatched. See _collective_fuzzy_experimental_check's own
    # docstring for why a per-party guard alone (the old production
    # circuit's only defense) is not sufficient. On any disagreement, this
    # is a coordinated reject - no party's training task is ever created,
    # not just the party that happened to detect the problem - and no
    # model result is produced.
    if matching_method == "fuzzy_experimental":
        alignment_ok, alignment_detail, approved_mapping_digest_by_client_id = _collective_fuzzy_experimental_check(
            client, client_org_ids, agg_org_ids, run_id, max_entities, database_by_client_id,
            expected_client_id, fuzzy_threshold, capacity_mode="dynamic" if is_automatic_capacity else "manual",
        )
        info(f"Central (train {architecture}, {privacy_mode}): fuzzy_experimental "
             f"collective alignment check {'PASSED' if alignment_ok else 'FAILED'} "
             f"- {alignment_detail}")
        if alignment_ok:
            # G01 fix: forward each party's OWN approved digest - never
            # read back from its artifact file, see
            # _collective_fuzzy_experimental_check's own docstring.
            client_kwargs["approved_mapping_digest_by_client_id"] = approved_mapping_digest_by_client_id
        if not alignment_ok:
            message = (
                "Collective alignment validation failed for "
                "matching_method='fuzzy_experimental' - no training task was "
                "dispatched to any party, and no model result was produced. "
                "See 'alignment_detail' for the non-PII per-party diagnostic "
                "(status/local_valid/intersection_size/id-set agreement)."
            )
            return {
                "summary": [
                    {"metric": "run_id", "value": run_id},
                    {"metric": "overall_status", "value": "alignment_error"},
                    {"metric": "architecture", "value": architecture},
                    {"metric": "dataset", "value": dataset},
                    {"metric": "privacy_mode", "value": privacy_mode},
                    {"metric": "matching_method", "value": matching_method},
                    {"metric": "message", "value": message},
                ],
                "run_id": run_id,
                "overall_status": "alignment_error",
                "architecture": architecture,
                "dataset": dataset,
                "privacy_mode": privacy_mode,
                "matching_method": matching_method,
                "fuzzy_threshold": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None,
                "aligned_count": None,
                "message": message,
                "alignment_detail": alignment_detail,
            }

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
                input_={"method": spec["agg_action"],
                        "kwargs": dict(agg_kwargs, recipient_key_sha256=researcher_key_sha256)},
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
        #
        # F02 fix: fuzzy_experimental is the ONE exception - the
        # collective check (already run, already required to pass before
        # this code is even reached) already ran this exact PSI
        # computation once and produced validated alignment artifacts
        # every worker/coordinator consumes directly (see
        # mpc_daemon_client_v2.py's _get_psi_result), with no live PSI
        # connection. Dispatching a SECOND aggregator psi_party_run here
        # would have those aggregators launch the circuit and wait for
        # client connections that are never coming, hanging until that
        # job's own timeout.
        if matching_method != "fuzzy_experimental":
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
    # secure mode (Stage 2): client subtasks report no predictions at all -
    # the circuit reveals only the collective validity bit to them - so
    # there is no cross-party prediction comparison; see the researcher-
    # output assembly below. non_secure (vanilla) mode's leak is
    # deliberately left as-is for now (already documented as
    # insecure-by-design), so its subtasks still return raw predictions,
    # compared directly as before.
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
    # G05 fix: named once, reused below for party_errors too - no
    # separate per-training aggregator task is dispatched for this
    # method/mode combination (see the dispatch block above), so
    # agg_org_ids simply has no entry in `results` and `tasks` for this
    # phase; both aggregators_ok and party_errors must treat that as
    # "nothing new to check here" rather than "didn't complete".
    agg_dispatched_this_phase = not (privacy_mode == "non_secure" and matching_method == "fuzzy_experimental")
    if not agg_dispatched_this_phase:
        # The aggregators already completed as part of the collective
        # check, a required gate before this code is ever reached.
        aggregators_ok = True
    else:
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
                {"metric": "dataset", "value": dataset},
                {"metric": "privacy_mode", "value": privacy_mode},
                {"metric": "matching_method", "value": matching_method},
                {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None},
                {"metric": "aligned_count", "value": 0},
                {"metric": "message", "value": message},
            ],
            "run_id": run_id,
            "overall_status": "no_matches",
            "architecture": architecture,
            "dataset": dataset,
            "privacy_mode": privacy_mode,
            "matching_method": matching_method,
            "fuzzy_threshold": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None,
            "aligned_count": 0,
            "predictions_agree": None,
            "clients_ok": True,
            "aggregators_ok": True,
            "party_errors": {},
            "message": message,
        }

    clients_ok = all(s == "complete" for s in client_statuses.values())

    for org_id in client_org_ids:
        r = results.get(org_id)
        if client_statuses[org_id] == "complete" and aligned_count is None:
            aligned_count = r.get("aligned_count")

    researcher_output = None
    if privacy_mode == "secure":
        # Stage 2 researcher-only output: no party reports predictions any
        # more, so there is nothing to cross-compare (predictions_agree is
        # not applicable). What is checked here is only that every party
        # completed and that all six envelopes arrived - three output-share
        # envelopes (one per computing party) and three preprocessing
        # envelopes (one per data holder). Every party only reaches
        # "complete" after the circuit's collective validity bit came back
        # 1, so a collectively rejected run never gets here with envelopes.
        # The envelopes are opaque to this function; whether they decrypt,
        # verify and reconstruct into a valid model is for the researcher's
        # tool alone to decide, so the status says "delivered", never
        # "success".
        shares = [results[o]["share_envelope"] for o in agg_org_ids
                  if (results.get(o) or {}).get("status") == "complete"
                  and isinstance(results[o].get("share_envelope"), dict)]
        preprocessing = [results[o]["preprocessing_envelope"] for o in client_org_ids
                         if client_statuses[o] == "complete"
                         and isinstance(results[o].get("preprocessing_envelope"), dict)]
        share_indexes = sorted((e.get("header") or {}).get("share_index", -1) for e in shares)
        client_ids = sorted((e.get("header") or {}).get("client_id", -1) for e in preprocessing)
        delivered = (clients_ok and aggregators_ok and share_indexes == [0, 1, 2] and client_ids == [0, 1, 2])
        predictions_agree = None
        overall_status = "encrypted_shares_delivered" if delivered else "incomplete"
        researcher_output = ({"status": "encrypted_shares_delivered", "run_id": run_id,
                              "shares": shares, "preprocessing": preprocessing,
                              "note": "encrypted to the researcher's key; reconstruct and validate locally with "
                                      "researcher_reconstruct.py - this result cannot confirm a usable model"}
                             if delivered else {"status": "not_delivered", "run_id": run_id})
    else:
        predictions_by_org = {}
        for org_id in client_org_ids:
            r = results.get(org_id)
            if client_statuses[org_id] == "complete":
                predictions_by_org[org_id] = r.get("predictions")

        predictions_agree = clients_ok and len(
            {tuple(p) for p in predictions_by_org.values() if p is not None}
        ) == 1

        # aggregators_ok computed earlier (needed there for the no_matches check too).
        overall_status = "success" if (clients_ok and aggregators_ok and predictions_agree) else "incomplete"

    # Sanitized per-party errors, same rationale as central.py's R05
    # fix: status + message only, surfaced for every party that did NOT
    # complete regardless of debug.
    #
    # G05 fix: only checks agg_org_ids when a per-training aggregator
    # task was actually dispatched this phase (see agg_dispatched_this_
    # phase above) - otherwise a successful non_secure/fuzzy_experimental
    # run reported 3 phantom "no result received" errors for aggregators
    # that were correctly never asked to do anything in this phase.
    party_errors = {}
    check_org_ids = list(client_org_ids) + (list(agg_org_ids) if agg_dispatched_this_phase else [])
    for org_id in check_org_ids:
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

    if privacy_mode == "secure":
        summary = _secure_summary(run_id, overall_status, architecture, dataset, matching_method, fuzzy_threshold,
                                  aligned_count, clients_ok, aggregators_ok, researcher_output, shares, preprocessing)
    else:
        summary = [
            {"metric": "run_id", "value": run_id},
            {"metric": "overall_status", "value": overall_status},
            {"metric": "architecture", "value": architecture},
            {"metric": "dataset", "value": dataset},
            {"metric": "privacy_mode", "value": privacy_mode},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None},
            {"metric": "aligned_count", "value": aligned_count},
            {"metric": "predictions_agree", "value": predictions_agree},
            {"metric": "clients_ok", "value": clients_ok},
            {"metric": "aggregators_ok", "value": aggregators_ok},
        ]
    output = {
        "summary": summary,
        "run_id": run_id,
        "overall_status": overall_status,
        "architecture": architecture,
        "dataset": dataset,
        "privacy_mode": privacy_mode,
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method in ("fuzzy", "fuzzy_experimental") else None,
        "aligned_count": aligned_count,
        "predictions_agree": predictions_agree,
        "clients_ok": clients_ok,
        "aggregators_ok": aggregators_ok,
        "party_errors": party_errors,
    }

    if researcher_output is not None:
        # Forwarded in the normal result - no debug needed.
        output["researcher_output"] = researcher_output

    if debug:
        # secure mode has no predictions anywhere (Stage 2); its per-party
        # results carry only statuses, counts and ciphertext. non_secure
        # mode is unchanged.
        if privacy_mode != "secure":
            output["predictions"] = next(
                (results[org_id].get("predictions") for org_id in client_org_ids
                 if results.get(org_id) and results[org_id].get("predictions") is not None),
                None,
            )
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
