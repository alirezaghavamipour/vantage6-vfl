import uuid

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info

from . import _psi_capacity


SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


@algorithm_client
def central(
    client: AlgorithmClient,
    client_org_ids: list,
    agg_org_ids: list,
    matching_method: str = "exact",
    fuzzy_threshold: int = 2,
    psi_capacity: int = None,
    psi_capacity_mode: str = "manual",
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

    F02: for matching_method="fuzzy" (the current production circuit),
    this function's result is DIAGNOSTIC ONLY, never a validated training
    alignment - matching_method="fuzzy" training is paused in
    central_train for exactly this reason. clients_agree here only
    compares each party's reported COUNT (intersection_size); it does not
    verify that the underlying accepted entities/alignment keys actually
    agree - a real run was observed with equal-looking counts that
    disagreed once the underlying alignment keys were compared per-row
    (see the F02 diagnostic in mpc_daemon_client_v2.py's
    _find_duplicate_align_keys). Use this to investigate alignment
    quality, not to certify it. matching_method="fuzzy_experimental" (the
    corrected unique-triple circuit) does not have this specific gap in
    its own matching logic, but calling it through this function still
    only gets you the same count-level clients_agree, not the stronger
    id-set check central_train's collective validation performs before
    training - see _collective_fuzzy_experimental_check in
    train_unified.py for that.

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
    uses the default (350). Schema discovery (report_schema_run) is
    dispatched to every client first, to learn the real raw row counts
    in the clear; rejection then happens clearly, before any PSI
    execution task is launched, if those counts exceed the selected
    capacity - never silently truncated. Ignored when
    psi_capacity_mode="automatic" (see below) - manual mode's own
    precompiled-circuit behavior is otherwise completely unchanged.

    When psi_capacity_mode="automatic" (Phase 2 Stage 4), the exact
    capacity is negotiated privately instead: schema discovery never
    learns row counts in this mode - the maximum row count across all
    clients IS revealed (that is the mechanism), but no individual
    party's count ever is. That maximum becomes the PSI capacity
    directly (any integer 1..1000). The corresponding circuit is then
    compiled on demand - first use at a given exact capacity takes
    noticeably longer (compilation time), later runs at the same exact
    capacity reuse the cached build - and every aggregator must
    independently report an identical build fingerprint and bytecode
    hash before PSI is dispatched; a missing/failed/disagreeing build
    aborts the whole run before any PSI execution starts. Initial
    support: matching_method="fuzzy_experimental", fuzzy_threshold=2
    only, up to 1000 raw rows per party.
    """
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

    # Ties every job this run dispatches - across all client and
    # aggregator hosts - back to this one orchestrated run, so anyone
    # debugging the daemon-bridge job files on disk (which have no other
    # way to tell which host's job belongs to which run) can find every
    # piece of it. Also logged below and returned in the summary.
    run_id = str(uuid.uuid4())

    info(f"Central: starting PSI run (run_id={run_id}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}) - clients={client_org_ids}, aggregators={agg_org_ids}")

    kwargs = {"matching_method": matching_method, "fuzzy_threshold": fuzzy_threshold, "run_id": run_id}

    # Phase 2 Stage 4: automatic capacity mode - psi_capacity == "auto"
    # negotiates max_entities privately (_psi_capacity.negotiate_automatic_
    # capacity, shared with central_train()) instead of computing it from
    # raw row counts, so schema discovery must never expose n_rows in
    # this mode.
    is_automatic_capacity = (
        matching_method == "fuzzy_experimental"
        and psi_capacity == _psi_capacity.AUTOMATIC_CAPACITY_SENTINEL
    )

    # Schema discovery: ask each party its own raw row count (read
    # straight from its CSV) before dispatching PSI, so the computing
    # parties can compile a correctly-shaped PSI circuit for this run's
    # actual data instead of a fixed assumed bound. This is safe to
    # auto-compute (unlike training's row bound): PSI's own bound only
    # needs to cover raw candidate counts, which don't require running
    # PSI first to learn (no chicken-and-egg problem). Rounded up to the
    # next multiple of 50 with at least 50 rows of headroom, so the
    # bound never lands exactly on any one party's true count. Skipped
    # entirely for automatic mode below - private row-count discovery
    # replaces it.
    schema_tasks = {
        org_id: client.task.create(
            input_={"method": "report_schema_run", "kwargs": {
                "run_id": run_id, "include_row_count": not is_automatic_capacity,
            }},
            organizations=[org_id], name=f"schema-psi-{org_id}",
        )["id"]
        for org_id in client_org_ids
    }
    schema_results = {org_id: client.wait_for_results(task_id=task_id)[0]
                       for org_id, task_id in schema_tasks.items()}

    if is_automatic_capacity:
        negotiation_ok, max_entities, negotiation_detail = _psi_capacity.negotiate_automatic_capacity(
            client, client_org_ids, agg_org_ids, run_id, fuzzy_threshold,
        )
        info(f"Central: automatic capacity negotiation {'PASSED' if negotiation_ok else 'FAILED'} "
             f"- {negotiation_detail}")
        if not negotiation_ok:
            message = (
                "Automatic capacity negotiation failed - no PSI task was dispatched to "
                "any party, and no result was produced. See 'negotiation_detail' for the "
                "non-PII diagnostic (no individual row counts are ever included)."
            )
            return {
                "summary": [
                    {"metric": "run_id", "value": run_id},
                    {"metric": "overall_status", "value": "capacity_negotiation_error"},
                    {"metric": "matching_method", "value": matching_method},
                    {"metric": "message", "value": message},
                ],
                "run_id": run_id,
                "overall_status": "capacity_negotiation_error",
                "matching_method": matching_method,
                "fuzzy_threshold": None,
                "intersection_size": None,
                "message": message,
                "negotiation_detail": negotiation_detail,
            }
        kwargs["capacity_mode"] = "dynamic"
        info(f"Central: automatic capacity negotiated max_entities={max_entities}")
    else:
        raw_row_counts = [schema_results[org_id]["n_rows"] for org_id in client_org_ids]
        max_entities = ((max(raw_row_counts) // 50) + 2) * 50
        if matching_method == "fuzzy_experimental":
            # Manual capacity selection (Phase 1, 2026-09-28): resolves
            # psi_capacity (explicit selection, or the default) against the
            # supported/validated set, and rejects clearly - before any
            # task is dispatched - if the real dataset's raw row count
            # exceeds it. See _psi_capacity.py.
            max_entities = _psi_capacity.resolve_capacity(psi_capacity, max_entities, raw_row_counts)
        info(f"Central: discovered raw row counts {raw_row_counts}, using PSI bound max_entities={max_entities}")
    kwargs["max_entities"] = max_entities

    # debug is threaded down to each client party's own subtask (not just
    # used below to filter central()'s own top-level output) since that
    # subtask's result is independently queryable in vantage6 regardless
    # of what central() itself returns. psi_party_run doesn't accept a
    # debug kwarg (aggregators never see matched names either way), so
    # this only applies to the client-share kwargs, not the shared dict
    # used for the aggregator loop below.
    client_kwargs = dict(kwargs, debug=debug)

    tasks = {}
    for org_id in client_org_ids:
        t = client.task.create(
            input_={"method": "psi_client_share", "kwargs": client_kwargs},
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
    #
    # R05 fix: agreement is only ever computed and reported over parties
    # that actually completed - a prior version silently dropped failed/
    # missing parties from BOTH the agreement check and the party count,
    # so e.g. 1 success + 2 failures could still report
    # clients_agree=True (trivially, comparing one value to itself). Now:
    # clients_ok/aggregators_ok require EVERY expected party to have
    # completed, and clients_agree is only ever True when clients_ok is
    # also True - "agree" now means "everyone finished AND everyone's
    # answer matched," never "the survivors happened to match."
    client_statuses = {org_id: (results.get(org_id) or {}).get("status") for org_id in client_org_ids}
    agg_statuses = {org_id: (results.get(org_id) or {}).get("status") for org_id in agg_org_ids}
    clients_ok = all(s == "complete" for s in client_statuses.values())
    aggregators_ok = all(s == "complete" for s in agg_statuses.values())

    client_answers = {
        org_id: results[org_id].get("intersection_size")
        for org_id in client_org_ids
        if client_statuses[org_id] == "complete"
    }
    intersection_size = next(iter(client_answers.values()), None)
    clients_agree = clients_ok and len(set(client_answers.values())) == 1

    overall_status = "success" if (clients_ok and aggregators_ok and clients_agree) else "incomplete"

    # Sanitized per-party errors: status + message only (never raw
    # stderr, which could in principle contain data this summary isn't
    # meant to expose) - available for every party that did NOT
    # complete, regardless of debug, since "which party failed and why"
    # is operational information the submitter needs to act on, not
    # sensitive row-level data.
    party_errors = {}
    for org_id in list(client_org_ids) + list(agg_org_ids):
        r = results.get(org_id)
        status = (r or {}).get("status")
        if status != "complete":
            party_errors[org_id] = {
                "status": status,
                "message": (r or {}).get("message", "no result received"),
            }

    info(f"Central: PSI {overall_status} - intersection_size={intersection_size}, "
         f"clients_agree={clients_agree}, clients_ok={clients_ok}, aggregators_ok={aggregators_ok}"
         + (f", errors={party_errors}" if party_errors else ""))

    # F02: label this result explicitly, in the result itself and not
    # just the docstring - "clients_agree" here is a COUNT match only,
    # not proof the underlying alignment keys agree (see docstring for
    # the real run where those diverged despite matching counts).
    diagnostic_note = None
    if matching_method == "fuzzy":
        diagnostic_note = (
            "DIAGNOSTIC ONLY, not a validated training alignment - "
            "clients_agree compares counts only, not the underlying "
            "alignment keys (see F02); matching_method='fuzzy' training "
            "is paused in central_train for this reason."
        )
    elif matching_method == "fuzzy_experimental":
        diagnostic_note = (
            "Count-level agreement only (clients_agree) - this function "
            "does not run the stronger collective id-set check "
            "central_train performs before training "
            "(_collective_fuzzy_experimental_check)."
        )

    output = {
        "summary": [
            {"metric": "run_id", "value": run_id},
            {"metric": "overall_status", "value": overall_status},
            {"metric": "matching_method", "value": matching_method},
            {"metric": "fuzzy_threshold", "value": fuzzy_threshold if matching_method == "fuzzy" else None},
            {"metric": "intersection_size", "value": intersection_size},
            {"metric": "clients_agree", "value": clients_agree},
            {"metric": "clients_ok", "value": clients_ok},
            {"metric": "aggregators_ok", "value": aggregators_ok},
            {"metric": "diagnostic_note", "value": diagnostic_note},
        ],
        "run_id": run_id,
        "overall_status": overall_status,
        "matching_method": matching_method,
        "fuzzy_threshold": fuzzy_threshold if matching_method == "fuzzy" else None,
        "intersection_size": intersection_size,
        "clients_agree": clients_agree,
        "clients_ok": clients_ok,
        "aggregators_ok": aggregators_ok,
        "party_errors": party_errors,
        "diagnostic_note": diagnostic_note,
    }

    if debug:
        output["client_results"] = {org_id: results.get(org_id) for org_id in client_org_ids}
        output["aggregator_results"] = {org_id: results.get(org_id) for org_id in agg_org_ids}

    return output
