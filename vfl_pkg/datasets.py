"""Dataset selection: maps a user-facing dataset choice to the per-party
node database label every task in one run must read, and validates the
algorithm/capacity settings against what that dataset needs - before any
task is dispatched.

Benchmark labels are registered under the SAME name on every party
(`<benchmark>_labelonly` / `<benchmark>_distributed`), so one selection
resolves to one label used identically for schema discovery, snapshot
retention, PSI and training.
"""
from . import _psi_capacity

DEFAULT_DATASET = "bcw_exact"

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
SUPPORTED_DATASETS = tuple(BENCHMARKS)

# Squared-error regression on a 0/1 target is valid, so classification
# datasets accept both; a continuous target cannot be trained as logistic.
_ALGORITHMS_BY_TASK = {"classification": ("logistic", "linear"), "regression": ("linear",)}


def validate_dataset(dataset):
    if dataset not in BENCHMARKS:
        raise ValueError(f"dataset={dataset!r} is not supported (choose one of {list(SUPPORTED_DATASETS)})")
    return BENCHMARKS[dataset]


def database_by_client_id(dataset, label_has_features):
    """CLIENT_ID -> node database label for training/schema discovery."""
    label = f"{dataset}_{'distributed' if label_has_features else 'labelonly'}"
    return {0: label, 1: label, 2: label}


def psi_database_label(dataset):
    """Label for standalone PSI. Identity columns are identical across a
    benchmark's two layouts, so either layout aligns the same rows;
    labelonly is used."""
    return f"{dataset}_labelonly"


def validate_psi_selection(dataset, matching_method, psi_capacity):
    """Checks that the dataset fits the selected manual fuzzy PSI capacity."""
    rows = validate_dataset(dataset)["rows_per_party"]
    if matching_method == "fuzzy_experimental" and psi_capacity != _psi_capacity.AUTOMATIC_CAPACITY_SENTINEL:
        selected = psi_capacity if psi_capacity is not None else _psi_capacity.DEFAULT_FUZZY_EXPERIMENTAL_CAPACITY
        if isinstance(selected, int) and not isinstance(selected, bool) and selected < rows:
            fitting = [c for c in _psi_capacity.SUPPORTED_FUZZY_EXPERIMENTAL_CAPACITIES if c >= rows]
            raise ValueError(
                f"dataset={dataset!r} has {rows} raw rows per party, but the selected PSI capacity is "
                f"{selected} - choose a capacity of at least {rows} (supported: {fitting}) or automatic mode"
            )


def validate_selection(dataset, algorithm, matching_method, psi_capacity, privacy_mode=None, n_samples=None):
    """Checks the dataset against the run's settings and returns the
    n_samples to use (secure training's aligned-row bound)."""
    meta = validate_dataset(dataset)
    if algorithm not in _ALGORITHMS_BY_TASK[meta["task"]]:
        raise ValueError(
            f"dataset={dataset!r} has a {meta['task']} target, which requires algorithm in "
            f"{list(_ALGORITHMS_BY_TASK[meta['task']])} (got algorithm={algorithm!r})"
        )
    validate_psi_selection(dataset, matching_method, psi_capacity)
    if privacy_mode == "secure" and n_samples is None:
        # Aligned rows can never exceed raw rows per party, so this bound
        # always suffices; pass a smaller n_samples to shrink the circuit.
        return meta["rows_per_party"]
    return n_samples
