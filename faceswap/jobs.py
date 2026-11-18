"""On-disk job storage and render progress tracking.

Layout:  <root>/<job_id>/input.<ext>, meta.json, faces/<a-d>/*.jpg, source_<a-d>.json, people.json,
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

from .config import FACE_SLOTS

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

    def face_dir(self, jid: str, slot: str) -> Path:
        if slot not in FACE_SLOTS or len(slot) != 1:
            raise ValueError(f"Unknown face '{slot}'.")
        return self.job_dir(jid) / "faces" / slot

    def face_paths(self, jid: str, slot: str) -> list[Path]:
        return sorted(self.face_dir(jid, slot).glob("*.*"))

    def target_dir(self, jid: str, slot: str) -> Path:
        """Photos of the person Face <slot> replaces."""
        if slot not in FACE_SLOTS or len(slot) != 1:
            raise ValueError(f"Unknown face '{slot}'.")
        return self.job_dir(jid) / "targets" / slot

    def target_paths(self, jid: str, slot: str) -> list[Path]:
        return sorted(self.target_dir(jid, slot).glob("*.*"))

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


    def unfinished_renders(self) -> list[str]:
        """Render ids whose saved state says they never finished (for resuming after a restart)."""
        out = []
        for st in self.root.glob("*/renders/*/state.json"):
            try:
                if json.loads(st.read_text()).get("state") in ("queued", "rendering", "finalizing"):
                    out.append(st.parent.name)
            except (OSError, ValueError):
                continue
        return sorted(out)


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


class FileProgress(MemoryProgress):
    """Progress store that also survives a restart: render state lives next to the render files.

    Keys: render:<rid> -> state.json, render:<rid>:chunk:<i> -> chunk_<i>.json,
          render:<rid>:cancel -> cancel flag, render:<rid>:live -> live.jpg
    """

    def __init__(self, store: "JobStore"):
        super().__init__()
        self.store = store

    def _path(self, key: str) -> Path | None:
        parts = key.split(":")
        if len(parts) < 2 or parts[0] != "render" or not RENDER_RE.match(parts[1]):
            return None
        try:
            d = self.store.render_dir(parts[1], create=True)
        except (NotFound, ValueError):
            return None
        if len(parts) == 2:
            return d / "state.json"
        if parts[2] == "chunk" and len(parts) == 4:
            return d / f"chunk_{int(parts[3]):03d}.json"
        if parts[2] == "cancel":
            return d / "cancel"
        if parts[2] == "live":
            return d / "live.jpg"
        return None

    def get(self, key, default=None):
        v = super().get(key)
        if v is not None:
            return v
        p = self._path(key)
        if p is None or not p.exists():
            return default
        try:
            v = p.read_bytes() if p.suffix == ".jpg" else True if p.name == "cancel" else json.loads(p.read_text())
        except (OSError, ValueError):
            return default
        super().put(key, v)
        return v

    def put(self, key, value):
        super().put(key, value)
        p = self._path(key)
        if p is None:
            return
        try:
            if p.suffix == ".jpg":
                p.write_bytes(value)
            elif p.name == "cancel":
                p.write_text("1")
            else:
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(value))
                tmp.replace(p)
        except OSError:
            pass
