"""One render at a time, on a background worker thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    project_id: str
    run: Callable[["Job"], dict]
    state: str = "queued"  # queued, running, done, failed, cancelled
    progress: dict = field(default_factory=dict)
    message: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_json(self) -> dict:
        return {"id": self.id, "project_id": self.project_id, "state": self.state, "progress": self.progress,
                "message": self.message, "result": self.result}


class JobQueue:
    def __init__(self):
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="render-worker", daemon=True)
        self._thread.start()

    def submit(self, project_id: str, run: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], project_id=project_id, run=run)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.state in ("queued", "running")]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel.set()
        if job.state == "queued":
            job.state = "cancelled"
            job.message = "Cancelled."
        return job

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job.cancel.is_set():
                continue
            job.state = "running"
            try:
                job.result = job.run(job) or {}
                job.state = "done"
            except Exception as exc:
                from .pipeline import Cancelled

                if isinstance(exc, Cancelled) or job.cancel.is_set():
                    job.state = "cancelled"
                    job.message = "Cancelled."
                else:
                    job.state = "failed"
                    job.message = str(exc) or exc.__class__.__name__
                    log.error("Render failed:\n%s", traceback.format_exc())
