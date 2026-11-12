"""Projects on disk: data/projects/<id>/project.json plus the media files."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ID_RE = re.compile(r"^[0-9a-f]{12}$")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = SAFE_NAME.sub("-", stem).strip("-._") or "video"
    return stem[:60]


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "projects"
        self.outputs = data_dir / "outputs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def dir(self, pid: str) -> Path:
        if not ID_RE.match(pid or ""):
            raise KeyError(pid)
        return self.root / pid

    def file(self, pid: str, name: str) -> Path:
        if "/" in name or "\\" in name or name.startswith("."):
            raise KeyError(name)
        return self.dir(pid) / name

    # --------------------------------------------------------------- projects
    def new(self, original_name: str) -> dict:
        pid = uuid.uuid4().hex[:12]
        d = self.root / pid
        d.mkdir(parents=True)
        return {
            "id": pid,
            "name": Path(original_name).name[:120] or "video",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video": None,
            "notes": [],
            "face": None,
            "references": {},
            "segments": [],
            "settings": {"output": "full", "max_quality": False, "det_size": 640, "default_mode": "everyone"},
        }

    def save(self, project: dict) -> None:
        with self._lock:
            d = self.dir(project["id"])
            tmp = d / "project.json.tmp"
            tmp.write_text(json.dumps(project, indent=2))
            tmp.replace(d / "project.json")

    def get(self, pid: str) -> dict:
        with self._lock:
            p = self.dir(pid) / "project.json"
            if not p.exists():
                raise KeyError(pid)
            return json.loads(p.read_text())

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            pj = d / "project.json"
            if pj.exists():
                try:
                    p = json.loads(pj.read_text())
                except ValueError:
                    continue
                if p.get("video"):
                    out.append({"id": p["id"], "name": p["name"], "created": p["created"]})
        return out

    def delete(self, pid: str) -> None:
        shutil.rmtree(self.dir(pid), ignore_errors=True)

    # ------------------------------------------------------------- embeddings
    def save_array(self, pid: str, name: str, arr: np.ndarray) -> None:
        np.save(self.file(pid, name), arr.astype(np.float32))

    def load_array(self, pid: str, name: str) -> np.ndarray:
        return np.load(self.file(pid, name))

    # ---------------------------------------------------------------- outputs
    def output_path(self, project: dict, mode: str) -> Path:
        stem = safe_stem(project["name"])
        tag = "clips" if mode == "clips" else "swapped"
        name = f"{stem}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        path = self.outputs / name
        n = 2
        while path.exists():
            path = self.outputs / f"{Path(name).stem}-{n}.mp4"
            n += 1
        return path

    def list_outputs(self) -> list[dict]:
        out = []
        for f in sorted(self.outputs.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name.endswith(".partial.mp4"):
                continue
            meta = {}
            side = f.with_suffix(".json")
            if side.exists():
                try:
                    meta = json.loads(side.read_text())
                except ValueError:
                    meta = {}
            out.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                **{k: meta[k] for k in ("project_id", "mode", "frames_swapped", "seconds") if k in meta},
            })
        return out

    def output_file(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.mp4", name or "") or name.endswith(".partial.mp4"):
            raise KeyError(name)
        p = self.outputs / name
        if not p.exists():
            raise KeyError(name)
        return p
