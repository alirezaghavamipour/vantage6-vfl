"""Shared, atomic job-publish / result-poll I/O for every vfl_pkg
wrapper module's own _run_job() (R04 from the 2026-09-26 code review).

Each module still builds its OWN job dict (they intentionally differ -
psi.py/train.py/the vanilla variants each need a different subset of
fields) and keeps its own timeout policy - only the actual FILE I/O is
shared here, so the atomicity fix and malformed-result handling lives
in exactly one place instead of being copy-pasted (and silently
drifting, per the review's own "Engineering improvement #2" note)
across what were 7 near-identical implementations.

The bug this fixes: `open(path, "w"); json.dump(...)` is not atomic - a
reader (the daemon, for a job; this module's own caller, for a result)
can observe a file that's been created but not finished being written.
Fix: write to a temp file in the same directory, then os.rename() -
atomic on the same filesystem, so a reader only ever sees the old state
(absent) or the complete new state, never a partial write.
"""
import json
import os
import tempfile
import time


def _atomic_write_json(path, obj):
    """Live-deployment bug found and fixed here: tempfile.mkstemp() always
    creates its file mode 0600, ignoring umask - unlike the plain
    open(path, "w") this replaced (which took the process's umask,
    typically world-readable). This function runs INSIDE the vantage6
    algorithm container as root (vantage6 nodes run Docker without user-
    namespace remapping), writing job files the HOST daemon (a normal,
    non-root user) must read back - a 0600 file owned by root is
    unreadable to that reader, so every job silently ended up quarantined
    instead of processed. chmod to 0644 before the rename restores
    cross-UID readability while keeping the atomicity fix."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        os.chmod(tmp_path, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# Deadline/heartbeat contract with the host daemons (daemons/job_control.py):
# every job carries an absolute deadline; the daemon stops its work
# CLEANUP_MARGIN before it and this side keeps waiting until RESULT_GRACE
# after it, so the container never gives up while the daemon still works.
RESULT_GRACE_SECONDS = 10
_SETTINGS = {"poll_interval": 1.0, "deadline": None, "readiness": False, "ready_poll": 0.1}
_DEADLINES = {}


def configure(poll_interval=None, deadline=None, readiness=None, ready_poll=None):
    """Per-container settings, set once by the task function from its kwargs."""
    if poll_interval is not None:
        _SETTINGS["poll_interval"] = min(max(float(poll_interval), 0.05), 5.0)
    if deadline is not None:
        _SETTINGS["deadline"] = float(deadline)
    if readiness is not None:
        _SETTINGS["readiness"] = bool(readiness)
    if ready_poll is not None:
        _SETTINGS["ready_poll"] = min(max(float(ready_poll), 0.05), 2.0)


def _bridge_dir(any_bridge_subdir):
    return os.path.dirname(os.path.abspath(any_bridge_subdir))


def _touch(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a"):
            os.utime(path, None)
    except OSError:
        pass


def publish_job(jobs_dir, job, timeout_s=None):
    """Atomically write job (must already contain 'job_id') to
    jobs_dir/<job_id>.json, stamped with its deadline (the run's shared
    deadline if configured, else now + timeout_s) and heartbeat contract."""
    os.makedirs(jobs_dir, exist_ok=True)
    now = time.time()
    job = dict(job)
    deadline = _SETTINGS["deadline"] if _SETTINGS["deadline"] is not None else (
        now + timeout_s if timeout_s else None)
    if deadline is not None:
        job["deadline"] = deadline
    job.update(heartbeat=True, published=now)
    if _SETTINGS["readiness"]:
        job.update(readiness=True, ready_poll=_SETTINGS["ready_poll"])
    _DEADLINES[job["job_id"]] = deadline
    _touch(os.path.join(_bridge_dir(jobs_dir), "heartbeats", job["job_id"]))
    path = os.path.join(jobs_dir, job["job_id"] + ".json")
    _atomic_write_json(path, job)
    return path


def _give_up(bridge, job_id):
    """Asks the daemon to stop this job and removes it if still queued."""
    _touch(os.path.join(bridge, "cancel", job_id))
    try:
        os.remove(os.path.join(bridge, "jobs", job_id + ".json"))
    except OSError:
        pass


def wait_for_result(results_dir, job_id, timeout_s, label="job"):
    """Poll results_dir/<job_id>.json until the job's deadline + grace (or
    timeout_s for a job published without one), refreshing its heartbeat,
    and return the parsed result dict, removing the file once read.
    Tolerates a transient malformed/torn read by treating it as "not ready
    yet". On giving up (or being interrupted) the daemon is told to stop
    the job; raises TimeoutError if no readable result arrives in time.
    """
    bridge = _bridge_dir(results_dir)
    heartbeat = os.path.join(bridge, "heartbeats", job_id)
    deadline = _DEADLINES.get(job_id)
    until = deadline + RESULT_GRACE_SECONDS if deadline is not None else time.time() + timeout_s
    result_path = os.path.join(results_dir, job_id + ".json")
    finished = False
    try:
        while time.time() < until:
            _touch(heartbeat)
            if os.path.exists(result_path):
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                except (json.JSONDecodeError, OSError, ValueError):
                    time.sleep(_SETTINGS["poll_interval"])
                    continue
                try:
                    os.remove(result_path)
                except OSError:
                    pass
                finished = True
                return result
            time.sleep(_SETTINGS["poll_interval"])
    finally:
        if not finished:
            _give_up(bridge, job_id)
        try:
            os.remove(heartbeat)
        except OSError:
            pass
    raise TimeoutError(f"{label}: host daemon did not respond to job {job_id} by its deadline")


def cancel_run(bridge, run_id):
    """Asks every daemon job of this run on this host to stop."""
    _touch(os.path.join(bridge, "cancel", run_id))
