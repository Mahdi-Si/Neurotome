"""In-process job runner: long attribution work runs on one worker thread and reports progress.

The API starts a job and returns at once; the browser follows it over SSE (``wait``) or by polling.
Cancellation is cooperative: a running job ends at its next ``progress()`` call, which raises ``Cancelled``.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class Cancelled(Exception):
    """Raised by the progress callback once cancel was requested."""


Progress = Callable[[float, str], None]  # progress(fraction 0..1, message); raises Cancelled when cancelled

TERMINAL = ("done", "error", "cancelled")
_ids = itertools.count(1)  # process-wide, so ids never repeat across runners


@dataclass
class Job:
    id: str
    kind: str
    run_id: str | None
    key: str | None  # dedup key (the cache file name); None = never dedup
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created: float = 0.0
    version: int = 0  # bumped on every change; SSE waits on it
    cancel_requested: bool = False

    def to_json(self) -> dict[str, Any]:
        keys = ("id", "kind", "run_id", "status", "progress", "message", "result", "error", "created")
        return {k: getattr(self, k) for k in keys}


# ponytail: one worker, in-memory jobs; persist or add GPU/CPU lanes when M4 captures need it.
class JobRunner:
    def __init__(self, keep: int = 100):
        self.keep = keep
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._fns: dict[str, Callable[[Progress], Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None

    def _new(self, kind: str, run_id: str | None, key: str | None) -> Job:
        job = Job(f"job_{next(_ids):06d}", kind, run_id, key, created=time.time())
        self._jobs[job.id] = job
        return job

    def _set(self, job: Job, **changes: Any) -> None:
        """Change a job and wake waiters; caller holds ``_cond``."""
        for k, v in changes.items():
            setattr(job, k, v)
        job.version += 1
        self._cond.notify_all()
        if job.status in TERMINAL:  # forget the oldest finished jobs
            done = [j.id for j in self._jobs.values() if j.status in TERMINAL]
            for jid in done[: max(0, len(done) - self.keep)]:
                del self._jobs[jid]

    def submit(self, kind: str, fn: Callable[[Progress], Any], run_id: str | None = None,
               key: str | None = None) -> Job:  # fmt: skip
        with self._cond:
            if key is not None:
                for j in self._jobs.values():
                    live = j.status in ("queued", "running") and not j.cancel_requested
                    if live and j.key == key and j.run_id == run_id:
                        return j
            job = self._new(kind, run_id, key)
            self._fns[job.id] = fn
            self._queue.put(job.id)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="neurotome-jobs", daemon=True)
                self._worker.start()
            return job

    def finished(self, kind: str, result: Any, run_id: str | None = None) -> Job:
        with self._cond:
            job = self._new(kind, run_id, None)
            self._set(job, status="done", progress=1.0, result=result)
            return job

    def get(self, job_id: str) -> Job:
        with self._cond:
            return self._jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        with self._cond:
            job = self._jobs[job_id]
            if job.status == "queued":
                self._fns.pop(job_id, None)
                self._set(job, status="cancelled", cancel_requested=True)
            elif job.status == "running":
                self._set(job, cancel_requested=True, message="Cancelling…")
            return job

    def wait(self, job_id: str, version: int, timeout: float) -> tuple[dict[str, Any], int]:
        with self._cond:
            job = self._jobs[job_id]
            self._cond.wait_for(lambda: job.version != version, timeout)
            return job.to_json(), job.version

    def _work(self) -> None:
        while True:
            jid = self._queue.get()
            with self._cond:
                job, fn = self._jobs.get(jid), self._fns.pop(jid, None)
                if job is None or fn is None or job.status != "queued":
                    continue
                self._set(job, status="running")

            def progress(frac: float, message: str, job: Job = job) -> None:
                with self._cond:
                    if job.cancel_requested:
                        raise Cancelled()
                    self._set(job, progress=min(max(float(frac), job.progress), 0.99), message=message)

            try:
                result = fn(progress)
            except Cancelled:
                changes = {"status": "cancelled"}
            except Exception as e:  # noqa: BLE001 - any failure is reported on the job
                changes = {"status": "error", "error": str(e) or type(e).__name__}
            else:
                changes = {"status": "done", "progress": 1.0, "result": result}
            with self._cond:
                self._set(job, **changes)
