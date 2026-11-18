"""Lead Role on Modal: web UI on a small CPU container, face work on autoscaling GPUs.

DEPLOY (one time, from any terminal - GitHub Codespaces works from a phone):
    pip install modal
    modal setup                  # prints a link - tap it, log in to Modal
    modal deploy modal_app.py    # first run builds the images (~10-15 min)
Modal prints a URL ending in .modal.run -> open it.

How a render gets fast: the trimmed section is cut into chunks and every chunk is
rendered on its own GPU at the same time (up to 12 for Fast), then the chunks are
stitched back together and the original audio is laid back on.

Optional environment variables (set before `modal deploy`):
    LEADROLE_GPU=A10G        GPU type for face work (A10G, L4, L40S, A100, H100...)
    LEADROLE_MAX_GPUS=16     cap on GPUs running at once
    LEADROLE_WARM_GPUS=0     keep N GPUs always on (no cold start, costs money while idle)
"""
import os
from pathlib import Path

import modal

from faceswap.config import MODEL_URLS

GPU = os.environ.get("LEADROLE_GPU", "A10G")
MAX_GPUS = int(os.environ.get("LEADROLE_MAX_GPUS", 16))
WARM_GPUS = int(os.environ.get("LEADROLE_WARM_GPUS", 0))
MODEL_DIR = "/models"
DATA = "/data"
JOBS = f"{DATA}/jobs"
HERE = Path(__file__).parent

NP = "numpy==1.26.4"
CV = "opencv-python-headless==4.10.0.84"

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "python-multipart", NP, CV)
    .env({"LEADROLE_STATIC": "/assets/static"})
    .add_local_python_source("faceswap")
    .add_local_dir(HERE / "faceswap" / "static", "/assets/static")
)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "curl", "unzip")
    .pip_install(NP, "cython", "setuptools", "wheel")
    .pip_install("insightface==0.7.3", NP, CV, extra_options="--no-build-isolation")
    .pip_install("onnxruntime-gpu==1.20.2", NP, CV)
    .run_commands(
        f"mkdir -p {MODEL_DIR}/buffalo_l",
        *[f"curl -fsSL --retry 3 -o {MODEL_DIR}/{name} {MODEL_URLS[name]}"
          for name in ("inswapper_128.onnx", "gfpgan_1.4.onnx", "bisenet_resnet_34.onnx")],
        f"curl -fsSL --retry 3 -o /tmp/buffalo_l.zip {MODEL_URLS['buffalo_l.zip']}"
        f" && unzip -j -o /tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d {MODEL_DIR}/buffalo_l"
        " && rm /tmp/buffalo_l.zip",
    )
    .env({"FACESWAP_MODEL_DIR": MODEL_DIR})
    .add_local_python_source("faceswap")
)

app = modal.App("lead-role")
volume = modal.Volume.from_name("lead-role-data", create_if_missing=True)
progress = modal.Dict.from_name("lead-role-progress", create_if_missing=True)


@app.cls(image=gpu_image, gpu=GPU, cpu=8.0, memory=16384, volumes={DATA: volume}, timeout=3600,
         scaledown_window=300, max_containers=MAX_GPUS, min_containers=WARM_GPUS)
class Worker:
    @modal.enter()
    def load(self):
        from faceswap.engine import FaceEngine
        from faceswap.jobs import JobStore

        self.engine = FaceEngine(MODEL_DIR)
        self.store = JobStore(JOBS)

    @modal.method()
    def ping(self) -> bool:
        return self.engine.gpu

    @modal.method()
    def faces(self, jid: str, slot: str) -> dict:
        from faceswap import pipeline as P

        volume.reload()
        out = P.analyze_faces(self.engine, self.store, jid, slot)
        volume.commit()
        return out

    @modal.method()
    def scan(self, jid: str, start: float, end: float) -> dict:
        from faceswap import pipeline as P

        volume.reload()
        out = P.scan_people(self.engine, self.store, jid, start, end)
        volume.commit()
        return out

    @modal.method()
    def preview(self, jid: str, t: float, cast: list, tier: str, vr: bool, frame: dict | None = None) -> dict:
        from faceswap import pipeline as P

        volume.reload()
        return P.preview(self.engine, self.store, jid, t, cast, tier, vr, frame)

    @modal.method()
    def render_chunk(self, rid: str, chunk: dict, req: dict) -> str:
        from faceswap import pipeline as P

        volume.reload()
        out = P.render_chunk(self.engine, self.store, rid, chunk, P.RenderRequest(**req), progress)
        volume.commit()
        return out


@app.function(image=web_image, volumes={DATA: volume}, cpu=2.0, memory=4096, timeout=4 * 3600)
def orchestrate(rid: str, req: dict, plan: dict):
    """Fan the chunks out to GPUs, wait, then stitch + mux the final video."""
    from faceswap import pipeline as P
    from faceswap.jobs import JobStore

    volume.reload()
    n = len(plan["chunks"])

    def run_chunks(chunks):
        for _ in Worker().render_chunk.map([rid] * n, chunks, [req] * n):
            pass

    P.run_render(JobStore(JOBS), progress, rid, P.RenderRequest(**req), plan, run_chunks,
                 sync=volume.reload, persist=volume.commit)


class ModalBackend:
    name = "modal"
    max_chunks = None  # each tier decides how many GPUs to use
    accelerator = f"{GPU} GPUs on Modal"
    progress = progress

    def _wait(self, fc, timeout: float) -> dict:
        try:
            return {"result": fc.get(timeout=timeout)}
        except TimeoutError:
            return {"pending": fc.object_id}

    def call(self, op: str, *args) -> dict:
        self.persist()
        return self._wait(getattr(Worker(), op).spawn(*args), 45)

    def poll(self, token: str) -> dict:
        from faceswap.jobs import NotFound

        if not token.startswith("fc-"):
            raise NotFound("task")
        return self._wait(modal.FunctionCall.from_id(token), 25)

    def start_render(self, rid, req, plan) -> None:
        from dataclasses import asdict

        orchestrate.spawn(rid, asdict(req), plan)

    def warm(self) -> None:
        Worker().ping.spawn()

    def sync(self) -> None:
        try:
            volume.reload()
        except Exception:  # noqa: BLE001 - fails while a download holds a file open; retry later
            pass

    def persist(self) -> None:
        volume.commit()

    def open_result(self, path: Path):
        try:
            it = iter(volume.read_file(str(path.relative_to(DATA))))
            first = next(it)
        except Exception:  # noqa: BLE001
            return None

        def stream():
            yield first
            yield from it

        return stream()


@app.function(image=web_image, volumes={DATA: volume}, cpu=1.0, memory=2048, timeout=3600,
              scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web():
    from faceswap.jobs import JobStore
    from faceswap.server import create_app

    store = JobStore(JOBS)
    return create_app(ModalBackend(), store)
