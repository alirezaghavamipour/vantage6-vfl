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
"""

SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES = (100, 350, 700, 1000)
DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY = 350


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
