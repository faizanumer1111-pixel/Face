"""The local web app. Run with ./run.sh (or python -m app.server)."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__, config
from .faces import FaceEngine, load_provider_settings
from .jobs import Job, JobQueue
from .pipeline import OUTPUT_MODES, RenderRequest, process_frame, render
from .projects import ProjectStore
from .segments import (MODES, SegmentError, check_ready, format_timecode, frame_to_seconds, from_json, normalize,
                       stats)
from .targets import iou, make_selector
from .video_io import (VideoError, VideoInfo, encode_jpeg, encoder_name, face_crop, load_image, prepare_source,
                       read_frame)

log = logging.getLogger("faceswap")

FACE_MISSING = "No face found in your photo — use a clear, front-facing shot."


class EngineHolder:
    """Loads the models once, in the background, as the app starts."""

    def __init__(self, factory: Callable[[], object]):
        self._factory = factory
        self._ready = threading.Event()
        self.engine = None
        self.error: str | None = None
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            self.engine = self._factory()
        except Exception as exc:
            self.error = str(exc)
            log.error("Couldn't load the face models: %s", exc)
        finally:
            self._ready.set()

    @property
    def state(self) -> str:
        if not self._ready.is_set():
            return "loading"
        return "ready" if self.engine is not None else "error"

    def get(self, timeout: float = 300):
        if not self._ready.wait(timeout):
            raise HTTPException(503, "The face models are still loading. Try again in a moment.")
        if self.engine is None:
            raise HTTPException(503, f"The face models didn't load: {self.error}")
        return self.engine


class FrameCache:
    def __init__(self, size: int = 12):
        self._d: OrderedDict = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key, load: Callable[[], np.ndarray]) -> np.ndarray:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        value = load()
        with self._lock:
            self._d[key] = value
            while len(self._d) > self._size:
                self._d.popitem(last=False)
        return value


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def create_app(data_dir: Path = config.DATA_DIR, models_dir: Path = config.MODELS_DIR,
               engine_factory: Callable[[], object] | None = None) -> FastAPI:
    data_dir.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(data_dir)
    jobs = JobQueue()
    frames = FrameCache()
    if engine_factory is None:
        def engine_factory():
            return FaceEngine(models_dir, load_provider_settings(data_dir))
    engines = EngineHolder(engine_factory)

    app = FastAPI(title="FaceSwap Studio", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.jobs = jobs
    app.state.engines = engines

    # Only answer requests addressed to this machine by name (blocks DNS rebinding).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.ALLOWED_HOSTS))

    @app.middleware("http")
    async def require_client_header(request: Request, call_next):
        # Other websites open in your browser can't add custom headers to requests
        # they send here, so they can't trigger uploads or renders.
        if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get(config.CLIENT_HEADER) != "1":
                return JSONResponse({"detail": "Blocked: this request didn't come from the FaceSwap Studio page."},
                                    status_code=403)
        return await call_next(request)

    @app.exception_handler(SegmentError)
    async def _segment_error(request: Request, exc: SegmentError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(VideoError)
    async def _video_error(request: Request, exc: VideoError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def _not_found(request: Request, exc: KeyError):
        return JSONResponse({"detail": "Not found."}, status_code=404)

    # ------------------------------------------------------------- helpers
    def info_of(project: dict) -> VideoInfo:
        if not project.get("video"):
            raise HTTPException(409, "Upload a video first.")
        return VideoInfo.from_json(project["video"])

    def work_path(project: dict) -> Path:
        return store.file(project["id"], "work.mp4")

    def get_frame(project: dict, n: int) -> np.ndarray:
        info = info_of(project)
        n = max(0, min(int(n), info.total_frames - 1))
        return frames.get((project["id"], n), lambda: read_frame(work_path(project), info, n))

    def segments_of(project: dict, info: VideoInfo):
        return from_json(project.get("segments", []), info.fps)

    def view(project: dict) -> dict:
        """Project JSON plus the derived bits the page needs."""
        out = dict(project)
        pid = project["id"]
        if project.get("video"):
            info = info_of(project)
            segs = segments_of(project, info)
            out["video_url"] = f"/api/projects/{pid}/video"
            out["stats"] = stats(segs, info.total_frames)
            out["segments"] = [
                {**s.to_json(), "start_tc": format_timecode(frame_to_seconds(s.start, info.fps)),
                 "end_tc": format_timecode(frame_to_seconds(s.end, info.fps))}
                for s in segs
            ]
        if project.get("face"):
            out["face"] = {**project["face"], "photo_url": f"/api/projects/{pid}/face/photo.jpg",
                           "crop_url": f"/api/projects/{pid}/face/crop.jpg?v={project['face'].get('selected', 0)}"}
        out["references"] = {rid: {**r, "crop_url": f"/api/projects/{pid}/references/{rid}.jpg"}
                             for rid, r in project.get("references", {}).items()}
        return out

    # --------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return FileResponse(config.WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")

    @app.get("/api/status")
    def status():
        eng = engines.engine
        return {
            "version": __version__,
            "engine": engines.state,
            "error": engines.error,
            "providers": getattr(eng, "providers", None),
            "encoder": encoder_name(False),
            "platform": platform.system().lower(),
            "busy": bool(jobs.active()),
            "modes": {m: m for m in MODES},
        }

    # ------------------------------------------------------------ projects
    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list()}

    @app.post("/api/projects")
    def create_project(video: UploadFile = File(...)):
        project = store.new(video.filename or "video.mp4")
        pid = project["id"]
        suffix = Path(video.filename or "").suffix.lower()[:8] or ".mp4"
        upload = store.file(pid, "upload" + suffix)
        with open(upload, "wb") as f:
            shutil.copyfileobj(video.file, f, length=4 * 1024 * 1024)
        try:
            info, notes = prepare_source(upload, store.file(pid, "work.mp4"))
        except VideoError:
            store.delete(pid)
            raise
        finally:
            upload.unlink(missing_ok=True)
        project["video"] = info.to_json()
        project["notes"] = notes
        store.save(project)
        return view(project)

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return view(store.get(pid))

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        store.get(pid)
        if any(j.project_id == pid for j in jobs.active()):
            raise HTTPException(409, "This project is rendering. Cancel the render first.")
        store.delete(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/video")
    def project_video(pid: str):
        project = store.get(pid)
        return FileResponse(work_path(project), media_type="video/mp4")

    @app.patch("/api/projects/{pid}/settings")
    def update_settings(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        s = project.setdefault("settings", {})
        if "output" in body:
            if body["output"] not in OUTPUT_MODES:
                raise HTTPException(422, "Output must be full or clips.")
            s["output"] = body["output"]
        if "max_quality" in body:
            s["max_quality"] = bool(body["max_quality"])
        if "det_size" in body:
            if int(body["det_size"]) not in (320, 480, 640):
                raise HTTPException(422, "Detection size must be 320, 480 or 640.")
            s["det_size"] = int(body["det_size"])
        if "default_mode" in body:
            if body["default_mode"] not in MODES:
                raise HTTPException(422, "Unknown mode.")
            s["default_mode"] = body["default_mode"]
        store.save(project)
        return view(project)

    # ---------------------------------------------------------------- face
    @app.post("/api/projects/{pid}/face")
    def upload_face(pid: str, photo: UploadFile = File(...)):
        project = store.get(pid)
        d = store.dir(pid)
        suffix = Path(photo.filename or "").suffix.lower()[:8] or ".jpg"
        raw = d / ("face_upload" + suffix)
        with open(raw, "wb") as f:
            shutil.copyfileobj(photo.file, f)
        try:
            img = load_image(raw, d)
        finally:
            raw.unlink(missing_ok=True)
            (d / "converted.png").unlink(missing_ok=True)
        engine = engines.get()
        faces = engine.detect(img, 640)
        if not faces:
            raise HTTPException(422, FACE_MISSING)
        embs = np.stack([engine.embed(img, f) for f in faces])
        selected = int(np.argmax([f.area for f in faces]))
        store.file(pid, "face_photo.jpg").write_bytes(encode_jpeg(img, 95))
        store.save_array(pid, "face_embeddings.npy", embs)
        project["face"] = {"width": img.shape[1], "height": img.shape[0],
                           "faces": [f.to_json() for f in faces], "selected": selected}
        _select_face(project, img, selected)
        store.save(project)
        return view(project)

    def _select_face(project: dict, img: np.ndarray, index: int) -> None:
        pid = project["id"]
        faces = project["face"]["faces"]
        if not 0 <= index < len(faces):
            raise HTTPException(422, "That face isn't in the photo.")
        embs = store.load_array(pid, "face_embeddings.npy")
        store.save_array(pid, "source_embedding.npy", embs[index])
        store.file(pid, "face_crop.jpg").write_bytes(encode_jpeg(face_crop(img, faces[index]["bbox"])))
        project["face"]["selected"] = index

    @app.post("/api/projects/{pid}/face/select")
    def select_face(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        if not project.get("face"):
            raise HTTPException(409, "Upload a face photo first.")
        import cv2

        img = cv2.imread(str(store.file(pid, "face_photo.jpg")))
        _select_face(project, img, int(body.get("index", -1)))
        store.save(project)
        return view(project)

    @app.get("/api/projects/{pid}/face/photo.jpg")
    def face_photo(pid: str):
        return FileResponse(store.file(pid, "face_photo.jpg"), media_type="image/jpeg")

    @app.get("/api/projects/{pid}/face/crop.jpg")
    def face_crop_img(pid: str):
        return FileResponse(store.file(pid, "face_crop.jpg"), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    # -------------------------------------------------------------- frames
    @app.get("/api/projects/{pid}/frames/{n}.jpg")
    def frame_jpeg(pid: str, n: int):
        project = store.get(pid)
        return Response(encode_jpeg(get_frame(project, n), 88), media_type="image/jpeg")

    @app.get("/api/projects/{pid}/frames/{n}/faces")
    def frame_faces(pid: str, n: int, det_size: int = 640):
        project = store.get(pid)
        frame = get_frame(project, n)
        faces = engines.get().detect(frame, det_size if det_size in (320, 480, 640) else 640)
        return {"frame": n, "faces": [f.to_json() for f in faces]}

    @app.post("/api/projects/{pid}/references")
    def create_reference(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        n = int(body["frame"])
        want = np.array(body["bbox"], dtype=np.float32)
        frame = get_frame(project, n)
        engine = engines.get()
        faces = engine.detect(frame, int(project["settings"].get("det_size", 640)))
        if not faces:
            raise HTTPException(422, "No faces on this frame. Move to a frame where the person is clearly visible.")
        best = max(faces, key=lambda f: iou(f.bbox, want))
        if iou(best.bbox, want) < 0.3:
            raise HTTPException(422, "Couldn't find that face again. Click directly on a face box.")
        emb = engine.embed(frame, best)
        rid = uuid.uuid4().hex[:8]
        store.save_array(pid, f"ref_{rid}.npy", emb)
        store.file(pid, f"ref_{rid}.jpg").write_bytes(encode_jpeg(face_crop(frame, best.bbox)))
        project.setdefault("references", {})[rid] = {"frame": n, "bbox": best.to_json()["bbox"]}
        seg_id = body.get("segment_id")
        if seg_id:
            for s in project.get("segments", []):
                if s.get("id") == seg_id:
                    s["ref"] = rid
                    s["mode"] = "one"
        store.save(project)
        return {"ref": rid, "project": view(project)}

    @app.get("/api/projects/{pid}/references/{rid}.jpg")
    def reference_img(pid: str, rid: str):
        return FileResponse(store.file(pid, f"ref_{rid}.jpg"), media_type="image/jpeg")

    # ------------------------------------------------------------ segments
    @app.put("/api/projects/{pid}/segments")
    def put_segments(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        info = info_of(project)
        segs = normalize(from_json(body.get("segments", []), info.fps), info.total_frames)
        project["segments"] = [s.to_json() for s in segs]
        store.save(project)
        return view(project)

    # ------------------------------------------------------------- preview
    def _source(project: dict) -> np.ndarray:
        if not project.get("face"):
            raise HTTPException(409, "Upload the face photo first.")
        return store.load_array(project["id"], "source_embedding.npy")

    @app.post("/api/projects/{pid}/preview")
    def preview(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        info = info_of(project)
        src = _source(project)
        mode = body.get("mode", "everyone")
        if mode not in MODES:
            raise HTTPException(422, "Unknown mode.")
        ref = None
        if mode == "one":
            rid = body.get("ref")
            if not rid or rid not in project.get("references", {}):
                raise HTTPException(422, "Pick the person first: click Pick person on this segment.")
            ref = store.load_array(pid, f"ref_{rid}.npy")
        engine = engines.get()
        frame = get_frame(project, int(body.get("frame", 0))).copy()
        before = encode_jpeg(frame, 88)
        selector = make_selector(mode, ref, float(body.get("threshold", 0.35)), float(info.fps))
        det_size = int(body.get("det_size", project["settings"].get("det_size", 640)))
        t0 = time.perf_counter()
        faces, targets = process_frame(engine, frame, selector, engine.latent(src), det_size)
        ms = (time.perf_counter() - t0) * 1000
        result = {"before": _data_url(before), "after": _data_url(encode_jpeg(frame, 88)),
                  "faces": len(faces), "swapped": len(targets), "ms": round(ms)}
        if mode == "one":
            result["similarities"] = [round(s, 2) for s in getattr(selector, "last_similarities", [])]
        return result

    # -------------------------------------------------------------- render
    @app.post("/api/projects/{pid}/render")
    def start_render(pid: str, body: dict = Body(default={})):
        project = store.get(pid)
        info = info_of(project)
        _source(project)
        segs = segments_of(project, info)
        check_ready(segs, set(project.get("references", {})))
        s = project["settings"]
        output = body.get("output", s.get("output", "full"))
        if output not in OUTPUT_MODES:
            raise HTTPException(422, "Output must be full or clips.")
        s["output"] = output
        s["max_quality"] = bool(body.get("max_quality", s.get("max_quality", False)))
        s["det_size"] = int(body.get("det_size", s.get("det_size", 640)))
        store.save(project)
        if any(j.project_id == pid for j in jobs.active()):
            raise HTTPException(409, "This project is already rendering.")
        job = jobs.submit(pid, lambda job: _run_render(job, pid))
        return {"job": job.to_json()}

    def _run_render(job: Job, pid: str) -> dict:
        project = store.get(pid)
        info = info_of(project)
        segs = segments_of(project, info)
        refs = {rid: store.load_array(pid, f"ref_{rid}.npy") for rid in project.get("references", {})
                if any(sg.ref == rid for sg in segs)}
        s = project["settings"]
        out = store.output_path(project, s["output"])
        engine = engines.get()
        req = RenderRequest(video_path=work_path(project), info=info, source_embedding=_source(project),
                            segments=segs, references=refs, output_path=out, output_mode=s["output"],
                            det_size=int(s.get("det_size", 640)), max_quality=bool(s.get("max_quality")))

        def progress(st):
            job.progress = st.to_json()

        t0 = time.time()
        result = render(req, engine, on_progress=progress, cancel=job.cancel,
                        log_path=store.dir(pid) / "last_render.log")
        meta = {"project_id": pid, "mode": s["output"], "frames_swapped": result.frames_swapped,
                "seconds": round(time.time() - t0, 1), "stats": result.to_json()}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        job.progress = result.to_json()
        return {"output": out.name, **meta}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.to_json() for j in jobs.active()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.get(job_id).to_json()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id).to_json()

    # ------------------------------------------------------------- outputs
    @app.get("/api/outputs")
    def list_outputs():
        return {"outputs": store.list_outputs()}

    @app.get("/api/outputs/{name}")
    def download_output(name: str):
        path = store.output_file(name)
        return FileResponse(path, media_type="video/mp4", filename=name)

    @app.post("/api/outputs/{name}/reveal")
    def reveal_output(name: str):
        path = store.output_file(name)
        if platform.system() != "Darwin":
            raise HTTPException(400, "Reveal in Finder only works on a Mac.")
        subprocess.run(["open", "-R", str(path)], check=False)
        return {"ok": True}

    return app


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start FaceSwap Studio")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if port_in_use(config.HOST, config.PORT):
        print(f"Port {config.PORT} is busy. FaceSwap Studio may already be running: {config.URL}")
        if not args.no_browser:
            webbrowser.open(config.URL)
        return 1
    import uvicorn

    app = create_app()
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(config.URL)).start()
    print(f"FaceSwap Studio is running at {config.URL} (only this Mac can open it). Press Ctrl+C to stop.")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
