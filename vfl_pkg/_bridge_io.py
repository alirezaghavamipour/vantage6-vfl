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


def publish_job(jobs_dir, job):
    """Atomically write job (must already contain 'job_id') to
    jobs_dir/<job_id>.json."""
    os.makedirs(jobs_dir, exist_ok=True)
    path = os.path.join(jobs_dir, job["job_id"] + ".json")
    _atomic_write_json(path, job)
    return path


def wait_for_result(results_dir, job_id, timeout_s, label="job"):
    """Poll results_dir/<job_id>.json for up to timeout_s seconds and
    return the parsed result dict, removing the file once read.
    Tolerates a transient malformed/torn read (should not occur given
    the daemon's own atomic write, but defensively retried rather than
    raised in case some other, non-atomic writer is ever in the loop)
    by treating it the same as "not ready yet." Raises TimeoutError if
    no readable result ever arrives within timeout_s.
    """
    result_path = os.path.join(results_dir, job_id + ".json")
    for _ in range(timeout_s):
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    result = json.load(f)
            except (json.JSONDecodeError, OSError, ValueError):
                time.sleep(1)
                continue
            try:
                os.remove(result_path)
            except OSError:
                pass
            return result
        time.sleep(1)
    raise TimeoutError(f"{label}: host daemon did not respond to job {job_id} within {timeout_s}s")
