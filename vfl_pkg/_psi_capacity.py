"""Shared fuzzy_experimental PSI capacity constants for central.py and
train_unified.py - the two orchestrator entry points that dispatch a
fuzzy_experimental PSI run and must agree on which capacities are
actually precompiled-and-validated on the aggregators.

This is the repo/vfl_pkg (Docker image) counterpart of
daemons/psi_fuzzy_capacity.py - the two can't literally share one
module (different deployment units: this one ships inside the
algorithm image, that one is scp'd directly to each host's
~/mpc-bridge), so the CAPACITY SET itself is kept here as the single
source of truth for this side, matching the daemon side's values
exactly. If the set of validated capacities ever changes, both files
need updating together - same drift risk circuit_generator.py's
GENERATOR_VERSION comment already documents for the training circuits,
just on the orchestrator side instead of the compile side.

Each of the 4 capacities was independently compiled (-R 64 -Z 3),
correctness-verified and performance-measured on isolated
infrastructure before being wired in - see the Phase 1 capacity
validation results (2026-09-28).

Phase 2 Stage 4: _negotiate_automatic_capacity (below) also lives here,
not in train_unified.py or central.py individually, specifically so
BOTH orchestrator entry points share the exact same negotiation logic -
central() (standalone PSI) and central_train() (training) must reach
identical build-agreement/rejection behavior for automatic mode, the
same way they already share resolve_capacity for manual mode.
"""
import json

SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES = (100, 350, 700, 1000)
DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY = 350

# k=1 addition: a SEPARATE constant from psi.py's/central.py's/
# train_unified.py's own SUPPORTED_FUZZY_THRESHOLDS=(1,2,3) - that one
# belongs to the OLD, still-disabled matching_method="fuzzy" path and
# is intentionally untouched here. fuzzy_experimental's own threshold
# support is narrower (only what has an actual compiled circuit) and
# was never validated at all before this addition - previously any
# fuzzy_threshold value silently passed straight through to the daemons
# for matching_method="fuzzy_experimental" with no upfront rejection.
SUPPORTED_FUZZY_EXPERIMENTAL_THRESHOLDS = (1, 2)
DEFAULT_FUZZY_EXPERIMENTAL_THRESHOLD = 2


def validate_fuzzy_experimental_threshold(fuzzy_threshold):
    """Rejects clearly, before any task is dispatched, an unsupported
    fuzzy_experimental threshold - mirrors resolve_capacity's own
    fail-fast pattern above. Called by central()/central_train() only
    when matching_method="fuzzy_experimental"; the old "fuzzy" method's
    own (1,2,3) validation is untouched and lives elsewhere."""
    if fuzzy_threshold not in SUPPORTED_FUZZY_EXPERIMENTAL_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold!r} is not a supported, "
            f"compiled fuzzy_experimental threshold (supported: "
            f"{sorted(SUPPORTED_FUZZY_EXPERIMENTAL_THRESHOLDS)})"
        )

# Phase 2 Stage 3: automatic capacity mode. psi_capacity="auto" bypasses
# the manual menu above entirely - the actual capacity is negotiated
# privately at run time (see train_unified.py's
# _negotiate_automatic_capacity) rather than chosen from a fixed set.
# DYNAMIC_MAX_CAPACITY mirrors daemons/dynamic_psi_capacity.py's own
# DYNAMIC_MAX_CAPACITY exactly (same "two mirrored copies, one per
# deployment unit" pattern as SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES
# above - see this module's own docstring) - a negotiated capacity
# above this is rejected here, before any compilation task is
# dispatched, matching ensure/compile_dynamic_capacity's own bound on
# the daemon side.
AUTOMATIC_CAPACITY_SENTINEL = "auto"
DYNAMIC_MAX_CAPACITY = 1000


def resolve_capacity(psi_capacity, computed_max_entities, raw_row_counts):
    """Resolves the actual PSI bound to use for a fuzzy_experimental
    run: the caller's explicit psi_capacity selection if given (must be
    one of SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES), else the default
    capacity - then checks the auto-computed bound (from discovered raw
    row counts) actually fits inside it. Raises ValueError (never
    silently clamps or truncates) if the selected capacity is
    unsupported or too small for the real data - the SAME rejection
    shape ensure_psi_compiled uses on the aggregator side, just
    surfaced here first. Note: schema-discovery tasks (report_schema_run)
    have ALREADY been dispatched to every client by the time this runs
    (raw_row_counts comes from their results) - this rejects before any
    PSI/training MPC execution task is launched, not before any task at
    all."""
    selected = psi_capacity if psi_capacity is not None else DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY
    if selected not in SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES:
        raise ValueError(
            f"psi_capacity={selected!r} is not a supported, benchmarked-and-"
            f"validated capacity (supported: "
            f"{sorted(SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES)}) - a different "
            f"capacity needs its own compiled-and-validated circuit first"
        )
    if computed_max_entities > selected:
        raise ValueError(
            f"fuzzy_experimental's raw dataset needs max_entities="
            f"{computed_max_entities} (from discovered row counts "
            f"{raw_row_counts}), but the selected capacity is {selected} - "
            f"choose a larger precompiled capacity or reduce the dataset"
        )
    return selected


def negotiate_automatic_capacity(client, client_org_ids, agg_org_ids, run_id, fuzzy_threshold,
                                  database_by_client_id=None):
    """Phase 2 Stage 3/4, automatic capacity mode (psi_capacity="auto"):
    private row-count discovery, then the build-agreement barrier - run
    to completion BEFORE any PSI or training task is dispatched. Same
    "one shared, coordinated reject" property as
    _collective_fuzzy_experimental_check in train_unified.py: any
    failure here means NO capacity is negotiated and nothing downstream
    (build or PSI) proceeds. This function only determines WHICH
    capacity to use and confirms it is actually compiled and agreed by
    all 3 aggregators - the PSI alignment itself still goes through
    each caller's own existing PSI dispatch afterward (with
    capacity_mode="dynamic"), reusing that established path rather than
    duplicating it. Shared by central() and central_train() (see this
    module's own docstring) - both must reach identical negotiation
    behavior.

    database_by_client_id: optional (central() has no per-architecture
    database selection; central_train() passes its own map, same
    convention as psi_client_share's own database_by_client_id) - None
    resolves to {} here, which count_discovery_client_run/report_schema
    already treat as "use the default database" (same as every other
    caller's own None-means-default convention elsewhere in this
    project).

    Individual row counts are never seen by this function or returned
    in its detail dict - count_discovery_client_run's own result only
    ever contains the CIRCUIT'S revealed (all_valid, max_count), the
    same collective reveal every client independently receives (see
    private_max_discovery.mpc's own module docstring).

    Returns (ok: bool, max_entities: int | None, detail: dict) - detail
    is safe to return directly in a caller's own result on failure (no
    PII, no per-party row counts)."""
    database_by_client_id = database_by_client_id or {}
    count_client_tasks = {
        org_id: client.task.create(
            input_={"method": "count_discovery_client_run", "kwargs": {
                "run_id": run_id, "database_by_client_id": database_by_client_id,
            }},
            organizations=[org_id], name=f"automatic-count-discovery-{org_id}",
        )["id"]
        for org_id in client_org_ids
    }
    count_agg_tasks = {
        org_id: client.task.create(
            input_={"method": "count_discovery_run", "kwargs": {"run_id": run_id}},
            organizations=[org_id], name=f"automatic-count-discovery-agg-{org_id}",
        )["id"]
        for org_id in agg_org_ids
    }
    count_results = {}
    for org_id, task_id in count_client_tasks.items():
        res = client.wait_for_results(task_id=task_id)
        count_results[org_id] = res[0] if res else None
    count_agg_results = {}
    for org_id, task_id in count_agg_tasks.items():
        res = client.wait_for_results(task_id=task_id)
        count_agg_results[org_id] = res[0] if res else None

    aggregators_ok = all(
        count_agg_results.get(org_id) and count_agg_results.get(org_id).get("status") == "complete"
        for org_id in agg_org_ids
    )
    clients_complete = all(
        count_results.get(org_id) and count_results.get(org_id).get("status") == "complete"
        for org_id in client_org_ids
    )
    if not (aggregators_ok and clients_complete):
        return False, None, {
            "stage": "count_discovery", "aggregators_ok": aggregators_ok, "clients_complete": clients_complete,
            "message": "private row-count discovery did not complete for every party "
                       "(missing participant or a connection failure) - no capacity negotiated",
        }

    reveals = {(count_results[org_id].get("all_valid"), count_results[org_id].get("max_count"))
               for org_id in client_org_ids}
    if len(reveals) != 1:
        return False, None, {
            "stage": "count_discovery",
            "message": "clients disagree on the revealed (all_valid, max_count) - "
                       "should be impossible for a correct MPC reveal",
        }
    all_valid, max_count = next(iter(reveals))
    if not all_valid:
        return False, None, {
            "stage": "count_discovery",
            "message": "private row-count discovery reported all_valid=False "
                       "(an invalid/empty/out-of-range dataset on at least one party)",
        }

    if not isinstance(max_count, int) or max_count < 1 or max_count > DYNAMIC_MAX_CAPACITY:
        return False, None, {
            "stage": "capacity_validation",
            "message": f"negotiated capacity={max_count!r} is outside the supported "
                       f"automatic-mode range [1, {DYNAMIC_MAX_CAPACITY}] - "
                       f"rejected before any compilation was dispatched",
        }

    build_tasks = {
        org_id: client.task.create(
            input_={"method": "prepare_dynamic_capacity_run", "kwargs": {
                "run_id": run_id, "max_entities": max_count, "fuzzy_threshold": fuzzy_threshold,
            }},
            organizations=[org_id], name=f"automatic-build-{org_id}",
        )["id"]
        for org_id in agg_org_ids
    }
    build_results = {}
    for org_id, task_id in build_tasks.items():
        res = client.wait_for_results(task_id=task_id)
        build_results[org_id] = res[0] if res else None

    missing = [org_id for org_id in agg_org_ids if not build_results.get(org_id)]
    if missing:
        return False, None, {
            "stage": "build_agreement", "missing": missing,
            "message": "no build response from one or more aggregators - aborting before any PSI execution",
        }
    failed = {
        org_id: (build_results[org_id].get("message") or build_results[org_id].get("stderr"))
        for org_id in agg_org_ids if build_results[org_id].get("status") != "complete"
    }
    if failed:
        return False, None, {
            "stage": "build_agreement", "failed": failed,
            "message": "dynamic capacity build failed on one or more aggregators - aborting before any PSI execution",
        }

    # Build-agreement barrier: every aggregator must report the SAME
    # fingerprint AND the SAME complete-artifact bytecode hash - not
    # just each individually succeeding. json.dumps(sort_keys=True)
    # makes fingerprint (a nested dict) hashable for set comparison.
    fingerprints = {org_id: build_results[org_id]["fingerprint"] for org_id in agg_org_ids}
    hashes = {org_id: build_results[org_id]["bytecode_hash"] for org_id in agg_org_ids}
    distinct_fp = {json.dumps(fp, sort_keys=True) for fp in fingerprints.values()}
    distinct_hash = set(hashes.values())
    if len(distinct_fp) != 1 or len(distinct_hash) != 1 or None in distinct_hash:
        return False, None, {
            "stage": "build_agreement", "hashes": hashes,
            "message": "aggregators disagree on the compiled build fingerprint/bytecode "
                       "hash - aborting before any PSI execution",
        }

    return True, max_count, {"stage": "complete", "max_entities": max_count, "bytecode_hash": next(iter(distinct_hash))}
