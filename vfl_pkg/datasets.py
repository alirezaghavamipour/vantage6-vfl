"""Dataset selection: maps a user-facing dataset choice to the per-party
node database label every task in one run must read, and validates the
algorithm/capacity settings against what that dataset needs - before any
task is dispatched.

Benchmark labels are registered under the SAME name on every party
(`<benchmark>_labelonly` / `<benchmark>_distributed`), so one selection
resolves to one label used identically for schema discovery, snapshot
retention, PSI and training. NODE_DEFAULT keeps the original per-party
`heart_vfl` / `heart_vfl_aggvfl` labels, whatever each node maps them to.
"""
from . import _psi_capacity

NODE_DEFAULT = "node_default"

# Row counts are per party (identical across parties and fixtures), which is
# what PSI capacity must cover. "regression" datasets have continuous targets.
_BENCHMARK_DATASETS = {
    "bcw": {"rows_per_party": 100, "task": "classification"},
    "diabetes": {"rows_per_party": 350, "task": "regression"},
    "credit": {"rows_per_party": 1000, "task": "classification"},
}
_FIXTURES = ("exact", "fuzzyk1", "fuzzyk2")

BENCHMARKS = {
    f"{name}_{fixture}": dict(meta, fixture=fixture)
    for name, meta in _BENCHMARK_DATASETS.items()
    for fixture in _FIXTURES
}
SUPPORTED_DATASETS = (NODE_DEFAULT,) + tuple(BENCHMARKS)

# Squared-error regression on a 0/1 target is valid, so classification
# datasets accept both; a continuous target cannot be trained as logistic.
_ALGORITHMS_BY_TASK = {"classification": ("logistic", "linear"), "regression": ("linear",)}


def validate_dataset(dataset):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset={dataset!r} is not supported (choose one of {list(SUPPORTED_DATASETS)})")
    return BENCHMARKS.get(dataset)


def database_by_client_id(dataset, label_has_features):
    """CLIENT_ID -> node database label for training/schema discovery."""
    if dataset == NODE_DEFAULT:
        return {
            0: "heart_vfl",
            1: "heart_vfl_aggvfl" if label_has_features else "heart_vfl",
            2: "heart_vfl_aggvfl" if label_has_features else "heart_vfl",
        }
    label = f"{dataset}_{'distributed' if label_has_features else 'labelonly'}"
    return {0: label, 1: label, 2: label}


def psi_database_label(dataset):
    """Label for standalone PSI. Identity columns are identical across a
    benchmark's two layouts, so either layout aligns the same rows;
    labelonly is used. None keeps the daemon's own default (heart_vfl)."""
    return None if dataset == NODE_DEFAULT else f"{dataset}_labelonly"


def validate_psi_selection(dataset, matching_method, psi_capacity):
    """Checks that a benchmark fits the selected manual fuzzy PSI capacity."""
    meta = validate_dataset(dataset)
    if meta is None:
        return
    rows = meta["rows_per_party"]
    if matching_method == "fuzzy_experimental" and psi_capacity != _psi_capacity.AUTOMATIC_CAPACITY_SENTINEL:
        selected = psi_capacity if psi_capacity is not None else _psi_capacity.DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY
        if isinstance(selected, int) and not isinstance(selected, bool) and selected < rows:
            fitting = [c for c in _psi_capacity.SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES if c >= rows]
            raise ValueError(
                f"dataset={dataset!r} has {rows} raw rows per party, but the selected PSI capacity is "
                f"{selected} - choose a capacity of at least {rows} (supported: {fitting}) or automatic mode"
            )


def validate_selection(dataset, algorithm, matching_method, psi_capacity, privacy_mode=None, n_samples=None):
    """Checks a benchmark selection against the run's settings and returns
    the n_samples to use (secure training's aligned-row bound). For
    node_default nothing is checked here and n_samples is returned as given."""
    meta = validate_dataset(dataset)
    if meta is None:
        return n_samples
    if algorithm not in _ALGORITHMS_BY_TASK[meta["task"]]:
        raise ValueError(
            f"dataset={dataset!r} has a {meta['task']} target, which requires algorithm in "
            f"{list(_ALGORITHMS_BY_TASK[meta['task']])} (got algorithm={algorithm!r})"
        )
    validate_psi_selection(dataset, matching_method, psi_capacity)
    rows = meta["rows_per_party"]
    if privacy_mode == "secure" and n_samples is None:
        # Aligned rows can never exceed raw rows per party, so this bound
        # always suffices; pass a smaller n_samples to shrink the circuit.
        return rows
    return n_samples
