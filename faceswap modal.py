"""
Video face swap (flat + VR side-by-side) on Modal.
Puts YOUR face onto the actor in ad footage.

LAUNCH (one time, from any terminal - GitHub Codespaces works from a phone):
    pip install modal
    modal setup                      # prints a link - tap it, log in to Modal
    modal deploy faceswap_modal.py   # first run builds the image, ~10-15 min
Modal prints a URL ending in .modal.run -> open it, upload a face photo + a video,
keep the page open, then tap Download.

VR: tick "VR side-by-side" for 180/360 SBS footage. Each eye is swapped on its
own and the two halves are stitched back together.
"""
import modal

MODEL_DIR = "/models"
INSIGHT_ROOT = "/root/.insightface"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("cython", "numpy<2", "setuptools", "wheel")
    .pip_install("insightface==0.7.3", extra_options="--no-build-isolation")
    .pip_install(
        "onnxruntime-gpu>=1.19",
        "opencv-python-headless",
        "huggingface_hub",
        "fastapi[standard]",
        "python-multipart",
    )
    .run_commands(
        # the swapper model
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d('ezioruan/inswapper_128.onnx', 'inswapper_128.onnx', local_dir='{MODEL_DIR}')\"",
        # face detector + recognizer pack
        "python -c \"from insightface.utils.storage import ensure_available; "
        f"ensure_available('models', 'buffalo_l', root='{INSIGHT_ROOT}')\"",
    )
)

app = modal.App("faceswap", image=image)
results = modal.Volume.from_name("faceswap-results", create_if_missing=True)


def _area(face):
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


@app.cls(gpu="A10G", timeout=3600, volumes={"/out": results}, scaledown_window=120)
class Swapper:
    @modal.enter()
    def load(self):
        import insightface

        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.analyzer = insightface.app.FaceAnalysis(name="buffalo_l", root=INSIGHT_ROOT, providers=prov)
        self.analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(f"{MODEL_DIR}/inswapper_128.onnx", providers=prov)

    def _swap(self, frame, src_face, largest_only):
        faces = self.analyzer.get(frame)
        if not faces:
            return frame
        if largest_only:
            faces = [max(faces, key=_area)]
        for f in faces:
            frame = self.swapper.get(frame, f, src_face, paste_back=True)
        return frame

    @modal.method()
    def run(self, face_bytes: bytes, video_bytes: bytes, vr_sbs: bool, largest_only: bool, job: str):
        import subprocess
        import cv2
        import numpy as np

        try:
            src_img = cv2.imdecode(np.frombuffer(face_bytes, np.uint8), cv2.IMREAD_COLOR)
            src_faces = self.analyzer.get(src_img)
            if not src_faces:
                raise ValueError("No face found in your photo. Use a clear, front-facing shot.")
            src_face = max(src_faces, key=_area)

            inp, tmp, out = f"/tmp/{job}_in.mp4", f"/tmp/{job}_video.mp4", f"/out/{job}.mp4"
            with open(inp, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(inp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if vr_sbs:
                    half = w // 2
                    left = self._swap(frame[:, :half].copy(), src_face, largest_only)
                    right = self._swap(frame[:, half:].copy(), src_face, largest_only)
                    frame = np.hstack([left, right])
                else:
                    frame = self._swap(frame, src_face, largest_only)
                writer.write(frame)
            cap.release()
            writer.release()

            # re-encode to H.264 and put the original audio back
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp, "-i", inp, "-map", "0:v", "-map", "1:a?",
                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", out],
                check=True, capture_output=True,
            )
        except Exception as e:  # surface the error on the status page
            with open(f"/out/{job}.error", "w") as f:
                f.write(str(e))
        results.commit()


@app.function(volumes={"/out": results})
@modal.asgi_app()
def web():
    import os
    import uuid
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    api = FastAPI()
    PAGE = ("<html><head><meta name='viewport' content='width=device-width'></head>"
            "<body style='font-family:sans-serif;max-width:480px;margin:40px auto;padding:0 16px'>{}</body></html>")

    @api.get("/", response_class=HTMLResponse)
    def home():
        return PAGE.format("""<h2>Face Swap</h2>
        <form action='/swap' method='post' enctype='multipart/form-data'>
          <p>Your face photo<br><input type='file' name='face' accept='image/*' required></p>
          <p>Video<br><input type='file' name='video' accept='video/*' required></p>
          <p><label><input type='checkbox' name='vr'> VR side-by-side (180/360)</label></p>
          <p><label><input type='checkbox' name='largest' checked> Only swap the biggest face in each frame</label></p>
          <button style='font-size:18px;padding:10px 24px'>Swap</button>
        </form>""")

    @api.post("/swap")
    async def swap(
        face: UploadFile = File(...),
        video: UploadFile = File(...),
        vr: str | None = Form(None),
        largest: str | None = Form(None),
    ):
        job = uuid.uuid4().hex[:10]
        Swapper().run.spawn(await face.read(), await video.read(), vr is not None, largest is not None, job)
        return RedirectResponse(f"/status/{job}", status_code=303)

    @api.get("/status/{job}", response_class=HTMLResponse)
    def status(job: str):
        results.reload()
        if os.path.exists(f"/out/{job}.error"):
            return PAGE.format(f"<h2>Failed</h2><p>{open(f'/out/{job}.error').read()}</p><a href='/'>Try again</a>")
        if os.path.exists(f"/out/{job}.mp4"):
            return PAGE.format(f"<h2>Done</h2><p><a href='/result/{job}' style='font-size:20px'>Download your video</a></p>"
                               "<a href='/'>Swap another</a>")
        return PAGE.format("<meta http-equiv='refresh' content='15'><h2>Working on it...</h2>"
                           "<p>This page refreshes itself every 15 seconds. Keep it open.</p>")

    @api.get("/result/{job}")
    def result(job: str):
        results.reload()
        return FileResponse(f"/out/{job}.mp4", media_type="video/mp4", filename=f"faceswap_{job}.mp4")

    return api
