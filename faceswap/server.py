"""HTTP API + static web UI. Backend-agnostic: the same app runs locally or on Modal."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline as P
from .config import FACE_SLOTS, MAX_SOURCE_PHOTOS, TIERS
from .jobs import JobStore, NotFound
from .media import MediaError, frame_jpeg, probe

STATIC = Path(os.environ.get("LEADROLE_STATIC", Path(__file__).parent / "static"))
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_DURATION = float(os.environ.get("LEADROLE_MAX_SECONDS", 20 * 60))


class Backend(Protocol):
    name: str
    accelerator: str
    max_chunks: int | None
    progress: object

    def call(self, op: str, *args) -> dict: ...      # {"result": ...} or {"pending": token}
    def poll(self, token: str) -> dict: ...
    def start_render(self, rid: str, req: P.RenderRequest, plan: dict) -> None: ...
    def warm(self) -> None: ...
    def sync(self) -> None: ...                       # see writes made by other machines
    def persist(self) -> None: ...                    # publish our writes to other machines
    def open_result(self, path: Path): ...            # fallback reader when a file isn't visible locally


def create_app(backend: Backend, store: JobStore) -> FastAPI:
    app = FastAPI(title="Lead Role", docs_url="/api/docs")

    @app.exception_handler(NotFound)
    async def _nf(_, exc):
        return JSONResponse({"error": f"That {exc.args[0] if exc.args else 'item'} doesn't exist (it may have expired)."},
                            status_code=404)

    @app.exception_handler(ValueError)
    async def _bad(_, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(MediaError)
    async def _media(_, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    def need_video(jid: str) -> None:
        if not (store.job_dir(jid) / "meta.json").exists():
            raise ValueError("Upload the ad video first.")

    def task(op: str, *args) -> dict:
        try:
            return backend.call(op, *args)
        except (ValueError, MediaError, NotFound):
            raise
        except Exception as e:  # noqa: BLE001
            # Remote exceptions keep their type on Modal; anything else is a server fault.
            raise HTTPException(500, f"{op} failed: {e}") from e

    # ------------------------------------------------------------ meta
    @app.get("/api/config")
    def config():
        return {"tiers": [t.public() for t in TIERS.values()], "backend": backend.name,
                "accelerator": backend.accelerator, "max_photos": MAX_SOURCE_PHOTOS, "face_slots": FACE_SLOTS,
                "parallel": backend.max_chunks, "max_seconds": MAX_DURATION}

    @app.post("/api/warm")
    def warm():
        backend.warm()
        return {"ok": True}

    # ------------------------------------------------------------ jobs
    @app.post("/api/jobs")
    def new_job():
        store.cleanup()
        jid = store.new_job()
        backend.persist()
        return {"job_id": jid}

    @app.put("/api/jobs/{jid}/video")
    async def upload_video(jid: str, file: UploadFile = File(...)):
        d = store.job_dir(jid)
        ext = Path(file.filename or "").suffix.lower()
        ext = ext if ext in VIDEO_EXTS else ".mp4"
        for old in list(d.glob("input.*")) + [d / "people.json", d / "meta.json"]:
            old.unlink(missing_ok=True)
        dest = d / f"input{ext}"
        with open(dest, "wb") as f:
            while chunk := await file.read(4 << 20):
                f.write(chunk)
        try:
            meta = await run_in_threadpool(probe, str(dest))
        except MediaError:
            dest.unlink(missing_ok=True)
            raise MediaError("That file isn't a video we can read. Try an MP4 or MOV.") from None
        if meta["duration"] > MAX_DURATION:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Videos up to {int(MAX_DURATION // 60)} minutes are supported. Trim it first.")
        store.write_json(jid, "meta.json", meta)
        await run_in_threadpool(backend.persist)
        return meta

    @app.post("/api/jobs/{jid}/faces/{slot}")
    async def upload_faces(jid: str, slot: str, files: list[UploadFile] = File(...)):
        """Replace the photos for one face (A-D) and check them."""
        d = store.face_dir(jid, slot)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for i, f in enumerate(files[:MAX_SOURCE_PHOTOS]):
            ext = Path(f.filename or "").suffix.lower()
            (d / f"{i}{ext if ext in IMAGE_EXTS else '.jpg'}").write_bytes(await f.read())
        await run_in_threadpool(backend.persist)
        return await run_in_threadpool(task, "faces", jid, slot)

    @app.delete("/api/jobs/{jid}/faces/{slot}")
    def remove_face(jid: str, slot: str):
        P.clear_face(store, jid, slot)
        backend.persist()
        return {"ok": True}

    @app.post("/api/jobs/{jid}/scan")
    def scan(jid: str, body: dict):
        need_video(jid)
        return task("scan", jid, float(body.get("start", 0)), float(body.get("end", 0)))

    @app.post("/api/jobs/{jid}/preview")
    def preview(jid: str, body: dict):
        need_video(jid)
        req = P.RenderRequest.parse(body)  # validates tier and cast
        return task("preview", jid, float(body.get("t", 0)), req.cast, req.tier, req.vr)

    @app.get("/api/tasks/{token}")
    def poll(token: str):
        return backend.poll(token)

    @app.get("/api/jobs/{jid}/frame")
    def frame(jid: str, t: float = 0.0, w: int = 240):
        jpg = frame_jpeg(str(store.video_path(jid)), t, max(64, min(w, 1280)))
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    # ------------------------------------------------------------ renders
    @app.post("/api/jobs/{jid}/render")
    def render(jid: str, body: dict):
        need_video(jid)
        if not body.get("consent"):
            raise ValueError("Confirm you have permission to use these faces before rendering.")
        req = P.RenderRequest.parse(body)
        plan = P.plan_render(store, jid, req, backend.max_chunks)
        rid = store.new_render_id(jid)
        store.render_dir(rid, create=True)
        backend.persist()
        backend.progress.put(f"render:{rid}", P.start_state(rid, req, plan))
        backend.start_render(rid, req, plan)
        return {"render_id": rid, "lanes": len(plan["chunks"]), "frames": plan["frames_total"]}

    @app.get("/api/renders/{rid}")
    def render_status(rid: str):
        st = P.render_status(backend.progress, rid)
        if st is None:
            raise NotFound("render")
        return st

    @app.get("/api/renders/{rid}/live.jpg")
    def live(rid: str):
        jpg = backend.progress.get(f"render:{rid}:live")
        if not jpg:
            return Response(status_code=204)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/renders/{rid}/video")
    def video(rid: str, request: Request, download: int = 0):
        path = store.render_dir(rid) / "final.mp4"
        if not path.exists():
            backend.sync()
        name = f"leadrole_{rid}.mp4" if download else None
        if path.exists():
            return ranged_file(path, request, name)
        stream = backend.open_result(path)  # visible remotely but not yet on this machine
        if stream is None:
            raise NotFound("video")
        headers = {"Content-Disposition": f'attachment; filename="{name}"'} if name else {}
        return StreamingResponse(stream, media_type="video/mp4", headers=headers)

    # ------------------------------------------------------------ UI
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def ranged_file(path: Path, request: Request, download_name: str | None = None) -> Response:
    """Serve a file with HTTP Range support so browsers can seek inside the video."""
    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    start, end, status = 0, size - 1, 200
    rng = request.headers.get("range", "")
    if rng.startswith("bytes="):
        first, _, last = rng[6:].split(",")[0].strip().partition("-")
        try:
            if first == "":
                start = max(0, size - int(last))
            else:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
        except ValueError:
            start, end = 0, size - 1
        if start > end or start >= size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)

    def body():
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=status, media_type="video/mp4", headers=headers)
