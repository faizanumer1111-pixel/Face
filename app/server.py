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
from .faces import FaceEngine, SwapOptions, load_provider_settings
from .jobs import Job, JobQueue
from .pipeline import OUTPUT_MODES, RenderRequest, latents_for, render
from .processor import FrameProcessor, selector_for
from .projects import FaceLibrary, ProjectStore, safe_stem
from .segments import (MODES, Segment, SegmentError, check_ready, format_timecode, frame_to_seconds, from_json,
                       normalize, stats)
from .targets import iou
from .video_io import (VideoError, VideoInfo, encode_jpeg, encoder_name, face_crop, load_image, prepare_source,
                       read_frame)
from .vr import FILENAME_TAGS, FORMAT_LABELS, FORMATS, Layout, eyes

log = logging.getLogger("faceswap")

FACE_MISSING = "No face found in your photo — use a clear, front-facing shot."
DET_SIZES = (320, 640, 960)
DEFAULT_SETTINGS = {
    "output": "full", "max_quality": False, "det_size": 640, "default_mode": "everyone",
    "enhance": 0.6, "occlusion": True, "color_match": True, "steady": True, "vr_format": "flat",
}


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


def _preview_jpeg(img: np.ndarray, longest: int = 2400) -> bytes:
    import cv2

    h, w = img.shape[:2]
    if max(h, w) > longest:
        s = longest / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(img, 88)


def create_app(data_dir: Path = config.DATA_DIR, models_dir: Path = config.MODELS_DIR,
               engine_factory: Callable[[], object] | None = None) -> FastAPI:
    data_dir.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(data_dir)
    library = FaceLibrary(data_dir)
    jobs = JobQueue()
    frames = FrameCache()
    if engine_factory is None:
        def engine_factory():
            return FaceEngine(models_dir, load_provider_settings(data_dir))
    engines = EngineHolder(engine_factory)

    app = FastAPI(title="FaceSwap Studio", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.library = library
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
            raise HTTPException(409, "Add a video first.")
        return VideoInfo.from_json(project["video"])

    def work_path(project: dict) -> Path:
        return store.file(project["id"], "work.mp4")

    def settings_of(project: dict) -> dict:
        return {**DEFAULT_SETTINGS, **project.get("settings", {})}

    def layout_of(project: dict) -> Layout:
        return Layout.of(settings_of(project)["vr_format"])

    def get_frame(project: dict, n: int) -> np.ndarray:
        info = info_of(project)
        n = max(0, min(int(n), info.total_frames - 1))
        return frames.get((project["id"], n), lambda: read_frame(work_path(project), info, n))

    def segments_of(project: dict, info: VideoInfo) -> list[Segment]:
        return from_json(project.get("segments", []), info.fps)

    def default_face(project: dict) -> str | None:
        ids = library.ids()
        fid = project.get("default_face")
        if fid in ids:
            return fid
        faces = library.list()
        return faces[0]["id"] if faces else None

    def face_json(meta: dict) -> dict:
        return {"id": meta["id"], "name": meta["name"], "crop_url": f"/api/faces/{meta['id']}/crop.jpg"}

    def view(project: dict) -> dict:
        """Project JSON plus the derived bits the page needs."""
        out = dict(project)
        pid = project["id"]
        out["settings"] = settings_of(project)
        out["default_face"] = default_face(project)
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
        out["references"] = {rid: {**r, "crop_url": f"/api/projects/{pid}/references/{rid}.jpg"}
                             for rid, r in project.get("references", {}).items()}
        return out

    def segment_at(project: dict, info: VideoInfo, frame: int) -> Segment | None:
        return next((s for s in segments_of(project, info) if s.start <= frame < s.end), None)

    def detect_frame(project: dict, frame: np.ndarray, det_size: int):
        """Faces in the frame (left eye for 3D/VR), with boxes in full-frame coordinates."""
        engine = engines.get()
        layout = layout_of(project)
        eye = eyes(frame, layout)[0]
        return engine, eye, engine.detect(eye, det_size)

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
            "features": getattr(eng, "features", {"enhance": False, "occlusion": False}),
            "encoder": encoder_name(False),
            "platform": platform.system().lower(),
            "busy": bool(jobs.active()),
            "formats": FORMAT_LABELS,
        }

    # ---------------------------------------------------------- face library
    @app.get("/api/faces")
    def list_faces():
        return {"faces": [face_json(m) for m in library.list()]}

    @app.post("/api/faces")
    def add_faces(photo: UploadFile = File(...)):
        scratch = data_dir / "tmp" / uuid.uuid4().hex[:8]
        scratch.mkdir(parents=True)
        try:
            suffix = Path(photo.filename or "").suffix.lower()[:8] or ".jpg"
            raw = scratch / ("photo" + suffix)
            with open(raw, "wb") as f:
                shutil.copyfileobj(photo.file, f)
            img = load_image(raw, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        engine = engines.get()
        faces = sorted(engine.detect(img, 640), key=lambda f: f.area, reverse=True)[:6]
        if not faces:
            raise HTTPException(422, FACE_MISSING)
        base = Path(photo.filename or "Face").stem.replace("_", " ").replace("-", " ").strip()[:28] or "Face"
        added = []
        for i, f in enumerate(faces):
            name = base if i == 0 else f"{base} {i + 1}"
            meta = library.add(name, encode_jpeg(face_crop(img, f.bbox, size=256)), engine.embed(img, f))
            added.append(face_json(meta))
        return {"added": added, "faces": [face_json(m) for m in library.list()]}

    @app.patch("/api/faces/{fid}")
    def rename_face(fid: str, body: dict = Body(...)):
        return face_json(library.rename(fid, str(body.get("name", ""))))

    @app.delete("/api/faces/{fid}")
    def delete_face(fid: str):
        library.delete(fid)
        return {"faces": [face_json(m) for m in library.list()]}

    @app.get("/api/faces/{fid}/crop.jpg")
    def face_crop_img(fid: str):
        return FileResponse(library.crop(fid), media_type="image/jpeg")

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
        # Keep the original name (VR tags often live in it), made safe for the disk.
        src = upload.with_name(f"src_{safe_stem(video.filename or 'video')}{suffix}")
        try:
            upload.rename(src)
            info, notes = prepare_source(src, store.file(pid, "work.mp4"))
        except VideoError:
            store.delete(pid)
            raise
        finally:
            src.unlink(missing_ok=True)
            upload.unlink(missing_ok=True)
        project["video"] = info.to_json()
        project["notes"] = notes
        project["settings"] = {**DEFAULT_SETTINGS, "vr_format": info.vr_format}
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
        proxy = store.file(pid, "proxy.mp4")
        return FileResponse(proxy if proxy.exists() else work_path(project), media_type="video/mp4")

    @app.patch("/api/projects/{pid}/settings")
    def update_settings(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        s = project.setdefault("settings", dict(DEFAULT_SETTINGS))
        if "output" in body:
            if body["output"] not in OUTPUT_MODES:
                raise HTTPException(422, "Output must be full or clips.")
            s["output"] = body["output"]
        for key in ("max_quality", "occlusion", "color_match", "steady"):
            if key in body:
                s[key] = bool(body[key])
        if "enhance" in body:
            s["enhance"] = float(min(1.0, max(0.0, float(body["enhance"]))))
        if "det_size" in body:
            if int(body["det_size"]) not in DET_SIZES:
                raise HTTPException(422, "Face detection must be 320, 640 or 960.")
            s["det_size"] = int(body["det_size"])
        if "default_mode" in body:
            if body["default_mode"] not in MODES:
                raise HTTPException(422, "Unknown mode.")
            s["default_mode"] = body["default_mode"]
        if "vr_format" in body:
            if body["vr_format"] not in FORMATS:
                raise HTTPException(422, "Unknown video format.")
            s["vr_format"] = body["vr_format"]
        if "default_face" in body:
            if body["default_face"] not in library.ids():
                raise HTTPException(422, "That face isn't in the library.")
            project["default_face"] = body["default_face"]
        store.save(project)
        return view(project)

    # -------------------------------------------------------------- frames
    @app.get("/api/projects/{pid}/frames/{n}.jpg")
    def frame_jpeg(pid: str, n: int):
        project = store.get(pid)
        return Response(_preview_jpeg(get_frame(project, n)), media_type="image/jpeg")

    @app.get("/api/projects/{pid}/frames/{n}/faces")
    def frame_faces(pid: str, n: int, det_size: int = 640):
        """Faces on a frame, and who is cast on each (for the casting board)."""
        project = store.get(pid)
        info = info_of(project)
        frame = get_frame(project, n)
        engine, eye, faces = detect_frame(project, frame, det_size if det_size in DET_SIZES else 640)
        seg = segment_at(project, info, n)
        out = []
        refs = {}
        if seg and seg.mode == "cast":
            for c in seg.cast:
                try:
                    refs[c["ref"]] = (store.load_array(pid, f"ref_{c['ref']}.npy"), c.get("face"))
                except (FileNotFoundError, KeyError):
                    continue
        default = default_face(project)
        for f in faces:
            item = f.to_json()
            cast = None
            if refs:
                emb = engine.embed(eye, f)
                best = max(refs.items(), key=lambda kv: float(np.dot(kv[1][0], emb)))
                if float(np.dot(best[1][0], emb)) >= seg.threshold:
                    cast = best[1][1] or default
            elif seg and seg.mode in ("everyone", "main"):
                cast = seg.face or default
            item["cast"] = cast
            out.append(item)
        if seg and seg.mode == "main" and out:
            biggest = max(range(len(out)), key=lambda i: faces[i].area)
            for i, item in enumerate(out):
                if i != biggest:
                    item["cast"] = None
        return {"frame": n, "faces": out, "segment": seg.id if seg else None,
                "mode": seg.mode if seg else None}

    def _make_reference(project: dict, n: int, bbox) -> str:
        pid = project["id"]
        want = np.array(bbox, dtype=np.float32)
        frame = get_frame(project, n)
        engine, eye, faces = detect_frame(project, frame, int(settings_of(project)["det_size"]))
        if not faces:
            raise HTTPException(422, "No faces on this frame. Move to a frame where the person is clearly visible.")
        best = max(faces, key=lambda f: iou(f.bbox, want))
        if iou(best.bbox, want) < 0.3:
            raise HTTPException(422, "Couldn't find that face again. Drop the face right on a face box.")
        emb = engine.embed(eye, best)
        rid = uuid.uuid4().hex[:8]
        store.save_array(pid, f"ref_{rid}.npy", emb)
        store.file(pid, f"ref_{rid}.jpg").write_bytes(encode_jpeg(face_crop(eye, best.bbox)))
        project.setdefault("references", {})[rid] = {"frame": n, "bbox": best.to_json()["bbox"]}
        return rid

    @app.post("/api/projects/{pid}/cast")
    def cast_face(pid: str, body: dict = Body(...)):
        """Put a library face on one person: the casting board's drag and drop."""
        project = store.get(pid)
        info = info_of(project)
        n = int(body["frame"])
        fid = body.get("face")
        if fid not in library.ids():
            raise HTTPException(422, "That face isn't in the library any more.")
        seg = segment_at(project, info, n)
        if seg is None:
            raise HTTPException(409, "Move the playhead into a segment first (or mark one with I and O).")
        rid = _make_reference(project, n, body["bbox"])
        new_emb = store.load_array(pid, f"ref_{rid}.npy")
        for s in project["segments"]:
            if s.get("id") != seg.id:
                continue
            if s.get("mode") != "cast":
                s["mode"], s["cast"] = "cast", []
            cast = s.setdefault("cast", [])
            # Same person cast again: replace their face instead of adding a second entry.
            for c in cast:
                try:
                    old = store.load_array(pid, f"ref_{c['ref']}.npy")
                except (FileNotFoundError, KeyError):
                    continue
                if float(np.dot(old, new_emb)) > 0.5:
                    c["face"], c["ref"] = fid, rid
                    break
            else:
                cast.append({"ref": rid, "face": fid})
        store.save(project)
        return view(project)

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
    def _faces_for(project: dict, segs: list[Segment]) -> dict[str, np.ndarray]:
        ids = {default_face(project)}
        for s in segs:
            ids.add(s.face)
            ids.update(c.get("face") for c in s.cast)
        have = library.ids()
        return {fid: library.embedding(fid) for fid in ids if fid and fid in have}

    def _refs_for(project: dict, segs: list[Segment]) -> dict[str, np.ndarray]:
        wanted = {c["ref"] for s in segs for c in s.cast}
        return {rid: store.load_array(project["id"], f"ref_{rid}.npy") for rid in wanted
                if rid in project.get("references", {})}

    @app.post("/api/projects/{pid}/preview")
    def preview(pid: str, body: dict = Body(...)):
        project = store.get(pid)
        info = info_of(project)
        s = settings_of(project)
        n = int(body.get("frame", 0))
        seg = segment_at(project, info, n)
        segs = [seg] if seg else []
        if seg is None and not default_face(project):
            raise HTTPException(409, "Add a face to put in first: drop a photo onto the faces panel.")
        check_ready(segs or [Segment(n, n + 1, s["default_mode"] if s["default_mode"] != "cast" else "everyone")],
                    set(project.get("references", {})), library.ids(), default_face(project))
        engine = engines.get()
        frame = get_frame(project, n).copy()
        before = _preview_jpeg(frame)
        selector = selector_for(seg, _refs_for(project, segs), default_face(project), float(info.fps),
                                mode=None if seg else (s["default_mode"] if s["default_mode"] != "cast" else "everyone"))
        proc = FrameProcessor(engine, layout_of(project), latents_for(engine, _faces_for(project, segs)),
                              SwapOptions.from_json(s), int(body.get("det_size", s["det_size"])), selector,
                              steady=False)
        t0 = time.perf_counter()
        r = proc.process(frame)
        ms = (time.perf_counter() - t0) * 1000
        result = {"before": _data_url(before), "after": _data_url(_preview_jpeg(frame)),
                  "faces": r.faces, "swapped": r.swapped, "ms": round(ms),
                  "mode": seg.mode if seg else s["default_mode"]}
        if r.similarities:
            result["similarities"] = [round(x, 2) for x in r.similarities]
        return result

    # -------------------------------------------------------------- render
    @app.post("/api/projects/{pid}/render")
    def start_render(pid: str, body: dict = Body(default={})):
        project = store.get(pid)
        info = info_of(project)
        segs = segments_of(project, info)
        check_ready(segs, set(project.get("references", {})), library.ids(), default_face(project))
        s = project.setdefault("settings", dict(DEFAULT_SETTINGS))
        output = body.get("output", settings_of(project)["output"])
        if output not in OUTPUT_MODES:
            raise HTTPException(422, "Output must be full or clips.")
        s["output"] = output
        if "max_quality" in body:
            s["max_quality"] = bool(body["max_quality"])
        if "det_size" in body and int(body["det_size"]) in DET_SIZES:
            s["det_size"] = int(body["det_size"])
        store.save(project)
        if any(j.project_id == pid for j in jobs.active()):
            raise HTTPException(409, "This project is already rendering.")
        job = jobs.submit(pid, lambda job: _run_render(job, pid))
        return {"job": job.to_json()}

    def _run_render(job: Job, pid: str) -> dict:
        project = store.get(pid)
        info = info_of(project)
        segs = segments_of(project, info)
        s = settings_of(project)
        out = store.output_path(project, s["output"])
        tag = FILENAME_TAGS.get(s["vr_format"])
        if tag and not out.stem.lower().endswith(tag.lower()):
            out = out.with_name(out.stem + tag + ".mp4")
        engine = engines.get()
        req = RenderRequest(video_path=work_path(project), info=info, faces=_faces_for(project, segs),
                            default_face=default_face(project), segments=segs, references=_refs_for(project, segs),
                            output_path=out, output_mode=s["output"], det_size=int(s["det_size"]),
                            max_quality=bool(s["max_quality"]), options=SwapOptions.from_json(s),
                            steady=bool(s["steady"]), vr_format=s["vr_format"])

        def progress(st):
            job.progress = st.to_json()

        t0 = time.time()
        result = render(req, engine, on_progress=progress, cancel=job.cancel,
                        log_path=store.dir(pid) / "last_render.log")
        meta = {"project_id": pid, "mode": s["output"], "frames_swapped": result.frames_swapped,
                "seconds": round(time.time() - t0, 1), "format": s["vr_format"], "stats": result.to_json()}
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

    @app.get("/api/outputs/{name}/thumb.jpg")
    def output_thumb(name: str):
        path = store.output_file(name)
        thumb = path.with_suffix(".thumb.jpg")
        if not thumb.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(path), "-frames:v", "1",
                            "-vf", "scale=320:-2", str(thumb)], capture_output=True)
        if not thumb.exists():
            raise KeyError(name)
        return FileResponse(thumb, media_type="image/jpeg")

    @app.post("/api/outputs/{name}/reveal")
    def reveal_output(name: str):
        path = store.output_file(name)
        if platform.system() != "Darwin":
            raise HTTPException(400, "Reveal in Finder only works on a Mac.")
        subprocess.run(["open", "-R", str(path)], check=False)
        return {"ok": True}

    @app.delete("/api/outputs/{name}")
    def delete_output(name: str):
        path = store.output_file(name)
        for p in (path, path.with_suffix(".json"), path.with_suffix(".thumb.jpg")):
            p.unlink(missing_ok=True)
        return {"outputs": store.list_outputs()}

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
