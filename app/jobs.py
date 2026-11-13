"""The render queue: renders run one at a time, in the order shown, on a background thread.

One at a time is deliberate: two renders at once would fight over the same GPU and
both finish later than if they ran back to back.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
KEEP_FINISHED = 40  # finished renders kept in the list


@dataclass
class Job:
    id: str
    project_id: str
    run: Callable[["Job"], dict]
    label: str = ""
    details: dict = field(default_factory=dict)  # what was queued: project name, faces, output...
    state: str = "queued"  # queued, running, done, failed, cancelled
    progress: dict = field(default_factory=dict)
    message: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_json(self, position: int | None = None) -> dict:
        return {"id": self.id, "project_id": self.project_id, "label": self.label, "details": self.details,
                "state": self.state, "progress": self.progress, "message": self.message, "result": self.result,
                "created": self.created, "started": self.started, "finished": self.finished,
                "position": position}


class JobQueue:
    def __init__(self):
        self._order: list[Job] = []  # queued + running, in run order
        self._finished: list[Job] = []  # newest first
        self._jobs: dict[str, Job] = {}
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="render-worker", daemon=True)
        self._thread.start()

    def submit(self, project_id: str, run: Callable[[Job], dict], label: str = "", details: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], project_id=project_id, run=run, label=label, details=details or {})
        with self._cv:
            self._jobs[job.id] = job
            self._order.append(job)
            self._cv.notify()
        return job

    def get(self, job_id: str) -> Job:
        with self._cv:
            return self._jobs[job_id]

    def active(self) -> list[Job]:
        with self._cv:
            return [j for j in self._order if j.state in ACTIVE]

    def position(self, job: Job) -> int | None:
        """1 for the render that's running (or next up), 2 for the one after, ..."""
        with self._cv:
            for i, j in enumerate(self._order):
                if j is job:
                    return i + 1
        return None

    def listing(self) -> list[dict]:
        with self._cv:
            out = [j.to_json(i + 1) for i, j in enumerate(self._order)]
            out += [j.to_json() for j in self._finished]
            return out

    def cancel(self, job_id: str) -> Job:
        with self._cv:
            job = self._jobs[job_id]
            job.cancel.set()
            if job.state == "queued":
                self._finish(job, "cancelled", "Cancelled.")
            return job

    def move(self, job_id: str, to: int) -> None:
        """Move a waiting render to place `to` (1-based) among the waiting ones."""
        with self._cv:
            job = self._jobs[job_id]
            if job.state != "queued":
                raise ValueError("Only renders that haven't started can be moved.")
            running = [j for j in self._order if j.state == "running"]
            waiting = [j for j in self._order if j.state == "queued" and j is not job]
            to = max(1, min(int(to), len(waiting) + 1))
            waiting.insert(to - 1, job)
            self._order = running + waiting

    def clear_finished(self) -> None:
        with self._cv:
            for j in self._finished:
                self._jobs.pop(j.id, None)
            self._finished = []

    def _finish(self, job: Job, state: str, message: str = "") -> None:
        # caller holds the lock
        job.state = state
        job.message = message or job.message
        job.finished = time.time()
        if job in self._order:
            self._order.remove(job)
        self._finished.insert(0, job)
        for old in self._finished[KEEP_FINISHED:]:
            self._jobs.pop(old.id, None)
        del self._finished[KEEP_FINISHED:]

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not any(j.state == "queued" for j in self._order):
                    self._cv.wait()
                job = next(j for j in self._order if j.state == "queued")
                job.state = "running"
                job.started = time.time()
            try:
                result = job.run(job) or {}
                with self._cv:
                    job.result = result
                    self._finish(job, "done")
            except Exception as exc:
                from .pipeline import Cancelled

                with self._cv:
                    if isinstance(exc, Cancelled) or job.cancel.is_set():
                        self._finish(job, "cancelled", "Cancelled.")
                    else:
                        log.error("Render failed:\n%s", traceback.format_exc())
                        self._finish(job, "failed", str(exc) or exc.__class__.__name__)
