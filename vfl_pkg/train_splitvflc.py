import json
import os
import time
import uuid

from vantage6.algorithm.tools.util import info

BRIDGE = "/mnt/mpcbridge"
JOBS_DIR = os.path.join(BRIDGE, "jobs")
RESULTS_DIR = os.path.join(BRIDGE, "results")

# Measured ~170s for the full 171-row / 13-feature / 200-epoch circuit -
# only modestly more than aggVFLc's ~130s, despite the extra Dense+ReLU
# hidden layer (compiled cost: 66M triples / 47.8K VM rounds, vs
# aggVFLc's 43M / 40K).
TRAIN_TIMEOUT = 2000

SUPPORTED_FUZZY_THRESHOLDS = (1, 2, 3)


def _run_job(action: str, matching_method: str = "exact", fuzzy_threshold: int = 2,
             run_id: str = None, schema: dict = None, algorithm: str = None,
             n_samples_bound: int = None, max_entities: int = None, debug: bool = False) -> dict:
    if matching_method == "fuzzy" and fuzzy_threshold not in SUPPORTED_FUZZY_THRESHOLDS:
        raise ValueError(
            f"fuzzy_threshold={fuzzy_threshold} is not supported "
            f"(choose one of {SUPPORTED_FUZZY_THRESHOLDS})"
        )
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "action": action,
        "method": matching_method,
        "fuzzy_threshold": fuzzy_threshold,
    }
    # run_id ties every job dispatched by one orchestrator call together
    # across however many hosts they land on - job_id alone is only
    # unique to this one job, with nothing connecting it to its siblings
    # from the same run. Optional/None for direct/manual invocation
    # outside an orchestrator.
    if run_id:
        job["run_id"] = run_id
    # schema: this run's actual circuit shape, discovered by
    # central_train from the real data - None falls back to the
    # known-working default shape (see ensure_circuit_compiled on the
    # computing party's daemon).
    if schema:
        job["schema"] = schema
    # algorithm: "logistic" (default) or "linear" - which circuit variant
    # this feature/label party should encode its data for.
    if algorithm:
        job["algorithm"] = algorithm
    # n_samples_bound: sent directly to this feature/label party (not
    # just to the computing parties via schema["n_samples"]), since it
    # independently needs to know how many padding rows to add.
    if n_samples_bound:
        job["n_samples_bound"] = n_samples_bound
    if max_entities:
        job["max_entities"] = max_entities
    if debug:
        job["debug"] = True
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(os.path.join(JOBS_DIR, job_id + ".json"), "w") as f:
        json.dump(job, f)
    info(f"Train (splitVFLc): submitted job {job_id} ({action}, method={matching_method}, "
         f"fuzzy_threshold={fuzzy_threshold}), waiting for host daemon...")

    result_path = os.path.join(RESULTS_DIR, job_id + ".json")
    for _ in range(TRAIN_TIMEOUT):
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            info(f"Train (splitVFLc): job {job_id} completed with status {result.get('status')}")
            return result
        time.sleep(1)
    raise TimeoutError(f"Train (splitVFLc): host daemon did not respond to job {job_id} within {TRAIN_TIMEOUT}s")


def train_client_run_splitvflc(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, algorithm: str = None, n_samples_bound: int = None,
                    max_entities: int = None, debug: bool = False):
    """Feature/label party: align rows and share this party's own
    columns into the splitVFLc training computation (Rep3 MPC -
    private).

    Same data contribution as train_client_run (aggVFLc) - only the
    circuit differs: a genuine trainable Dense+ReLU hidden layer plus a
    Dense+Sigmoid output layer, instead of a single fixed logistic
    regression layer. Internally reuses the same Private PSI logic as
    every other function here to independently re-derive the aligned
    row set.

    algorithm/n_samples_bound/max_entities: see train_client_run (train.py).
    """
    return _run_job("train_client_run_splitvflc", matching_method, fuzzy_threshold, run_id,
                     algorithm=algorithm, n_samples_bound=n_samples_bound, max_entities=max_entities,
                     debug=debug)


def train_party_run_splitvflc(matching_method: str = "exact", fuzzy_threshold: int = 2,
                    run_id: str = None, schema: dict = None, max_entities: int = None):
    """Computing party: run this party's role in the Rep3 splitVFLc
    training computation (trainable-module vertical FL; the label party
    contributes no features of its own, same as aggVFLc). This
    computing party never sees any feature, label, or prediction - only
    its own secret share of the computation.

    schema: this run's discovered circuit shape - None uses the
    known-working default shape. max_entities: PSI's own row-count
    bound - None uses this daemon's own local default.

    This circuit is ONE joint Dense+ReLU hidden layer over every
    party's concatenated features (a single shared weight matrix) -
    splitVFLc's non_secure baseline mirrors this exact model in
    plaintext (see train_splitvflc_vanilla.py), so accuracy/predictions
    are directly comparable across privacy_mode for this architecture.
    """
    return _run_job("train_party_run_splitvflc", matching_method, fuzzy_threshold, run_id, schema, max_entities=max_entities)
