"""On-disk job storage and render progress tracking.

Layout:  <root>/<job_id>/input.<ext>, meta.json, faces/*.jpg, source.json, people.json,
         <root>/<job_id>/renders/<render_id>/chunk_000.mp4 ... final.mp4
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

JOB_RE = re.compile(r"^[a-f0-9]{12}$")
RENDER_RE = re.compile(r"^([a-f0-9]{12})-([a-f0-9]{6})$")


class NotFound(KeyError):
    pass


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ids
    def new_job(self) -> str:
        jid = uuid.uuid4().hex[:12]
        (self.root / jid / "faces").mkdir(parents=True)
        return jid

    def job_dir(self, jid: str) -> Path:
        if not JOB_RE.match(jid or "") or not (self.root / jid).is_dir():
            raise NotFound("job")
        return self.root / jid

    @staticmethod
    def new_render_id(jid: str) -> str:
        return f"{jid}-{uuid.uuid4().hex[:6]}"

    def render_dir(self, rid: str, create: bool = False) -> Path:
        m = RENDER_RE.match(rid or "")
        if not m:
            raise NotFound("render")
        d = self.job_dir(m.group(1)) / "renders" / rid
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # files
    def video_path(self, jid: str) -> Path:
        d = self.job_dir(jid)
        for p in d.glob("input.*"):
            return p
        raise NotFound("video")

    def face_paths(self, jid: str) -> list[Path]:
        return sorted((self.job_dir(jid) / "faces").glob("*.*"))

    def read_json(self, jid: str, name: str, default=None):
        p = self.job_dir(jid) / name
        if not p.exists():
            if default is not None:
                return default
            raise NotFound(name)
        return json.loads(p.read_text())

    def write_json(self, jid: str, name: str, data) -> None:
        p = self.job_dir(jid) / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)

    def cleanup(self, max_age_hours: float = 72) -> int:
        """Delete jobs older than max_age_hours. Returns how many were removed."""
        cutoff, n = time.time() - max_age_hours * 3600, 0
        for d in self.root.iterdir():
            if d.is_dir() and JOB_RE.match(d.name) and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n


class MemoryProgress:
    """Progress store for single-process runs. The Modal backend swaps in a modal.Dict."""

    def __init__(self):
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
