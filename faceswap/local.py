"""Run everything in one process (your own PC: NVIDIA GPU if available, else CPU)."""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from . import pipeline as P
from .engine import FaceEngine
from .jobs import FileProgress, JobStore, NotFound

OPS = {"faces": P.analyze_faces, "target": P.analyze_target, "scan": P.scan_people, "preview": P.preview}


class LocalBackend:
    name = "local"
    max_chunks = 1  # one GPU/CPU: splitting a render would not make it faster

    def __init__(self, store: JobStore, model_dir: str, wait_seconds: float = 25):
        self.store, self.model_dir, self.wait = store, model_dir, wait_seconds
        self.progress = FileProgress(store)
        self._engine: FaceEngine | None = None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(2)
        self._render_pool = ThreadPoolExecutor(1)  # renders queue up one after another
        self._tasks: dict = {}
        self._renders: dict = {}
        self.resume_unfinished()

    def resume_unfinished(self) -> None:
        """Pick up renders that were queued or running when the server last stopped."""
        for rid in self.store.unfinished_renders():
            st = self.progress.get(f"render:{rid}")
            if not st or "plan" not in st:
                st and self.progress.put(f"render:{rid}", dict(st, state="error", error="The server restarted before this render finished. Render it again."))
                continue
            try:
                req = P.RenderRequest(**st["request"])
            except (TypeError, ValueError):
                continue
            st["state"] = "queued"
            self.progress.put(f"render:{rid}", st)
            print(f"  resuming render {rid}", flush=True)
            self.start_render(rid, req, st["plan"])

    @property
    def engine(self) -> FaceEngine:
        with self._lock:
            if self._engine is None:
                self._engine = FaceEngine(self.model_dir)
            return self._engine

    @property
    def accelerator(self) -> str:
        if self._engine is None:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                return "GPU (CUDA)"
            if "CoreMLExecutionProvider" in avail:
                return "Apple silicon (CoreML)"
            return "CPU"
        return self._engine.device

    def warm(self) -> None:
        self._pool.submit(lambda: self.engine)

    def call(self, op: str, *args) -> dict:
        fut = self._pool.submit(lambda: OPS[op](self.engine, self.store, *args))
        try:
            return {"result": fut.result(timeout=self.wait)}
        except FutureTimeout:
            token = uuid.uuid4().hex
            self._tasks[token] = fut
            return {"pending": token}

    def poll(self, token: str) -> dict:
        fut = self._tasks.get(token)
        if fut is None:
            raise NotFound("task")
        try:
            result = fut.result(timeout=self.wait)
        except FutureTimeout:
            return {"pending": token}
        self._tasks.pop(token, None)
        return {"result": result}

    def start_render(self, rid: str, req: P.RenderRequest, plan: dict) -> None:
        def run_chunks(chunks):
            for c in chunks:
                P.render_chunk(self.engine, self.store, rid, c, req, self.progress)

        self._renders[rid] = self._render_pool.submit(P.run_render, self.store, self.progress, rid, req, plan, run_chunks)

    def cancel(self, rid: str):
        fut = self._renders.get(rid)
        if fut is not None:
            fut.cancel()  # drops it from the queue if it hasn't started
        return P.cancel_render(self.progress, rid)

    def sync(self) -> None:
        pass

    def persist(self) -> None:
        pass

    def open_result(self, path):
        return None
