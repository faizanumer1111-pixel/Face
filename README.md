# Lead Role

Put yourself in the ad. Drop in a photo of your face and an ad video, pick the person in the
video you're replacing, choose the part of the video you want, and render it at one of three
quality levels.

## What you can do

| | |
|---|---|
| **Your face** | Drop 1 to 5 photos, or take one with your camera. Several angles are averaged into one identity for a closer likeness. Photos of someone else are spotted and left out. |
| **Pick the section** | Drag the green in/out handles on the filmstrip, or press `I` and `O` at the playhead. Only that section is rendered, and its audio is kept. |
| **Pick the person** | **Find people** scans the section and lists everyone in it with how much of the section they're on screen. Click the one you're replacing; green ticks on the timeline show where they appear. Only that person is swapped. **Every face** is there too. |
| **Preview a frame** | Swaps the frame under the playhead in a few seconds and shows a draggable before/after split, before you spend time on a render. |
| **Render** | Shows each GPU's progress, a live frame from the render, time left, and how much of the section your face was found in. |
| **Compare and download** | A slider over the playing result shows the original underneath. Every render is kept as a take so you can compare Fast against Premium. |
| **VR** | Tick *VR side-by-side* for 180/360 stereo footage; each eye is swapped and tracked separately. |

Shortcuts: `Space` play/pause, `←`/`→` one frame (`Shift` for one second), `I`/`O` set in/out, `P` preview, `Esc` back to the video.

## The three quality levels

| | Fast | Better | Premium |
|---|---|---|---|
| Face swap | 128 px | 128 px | 256 px (4 sub-pixel passes) |
| Face restoration (GFPGAN) | none | 70% blend | 85% blend |
| Landmark smoothing | none | One-Euro filter | stronger One-Euro filter |
| Hands, mics and hair in front of the face | covered | covered | kept in front (face-parsing mask) |
| Face detector size | 480 | 640 | 800 (finds smaller faces) |
| Encode | x264 veryfast, CRF 20 | x264 medium, CRF 17 | x264 slow, CRF 14 |
| GPUs per render (Modal) | up to 12 | up to 10 | up to 8 |

All three are tuned in [`faceswap/config.py`](faceswap/config.py).

**Why renders are fast.** The section is cut into chunks and every chunk renders on its own GPU
at the same time, then the chunks are stitched back together frame-accurately and the original
audio is laid back on. Inside each GPU, face detection for upcoming frames runs in parallel with
the swap of the current frame, decoding and encoding run on their own threads, and blending
touches only the pixels around the face instead of the whole frame. Dropping a file in also
starts a GPU warming up, so it's usually ready by the time you press Render.

## Deploy on Modal (recommended)

One time, from any terminal (GitHub Codespaces works from a phone):

```bash
pip install modal
modal setup                 # prints a link - open it and log in to Modal
modal deploy modal_app.py   # first run builds the images, about 10-15 minutes
```

Modal prints a URL ending in `.modal.run`. Open it, and keep that page open while a render runs.

Optional settings, set before `modal deploy`:

| Variable | Default | What it does |
|---|---|---|
| `LEADROLE_GPU` | `A10G` | GPU type for face work (`L4`, `L40S`, `A100`, `H100`...) |
| `LEADROLE_MAX_GPUS` | `16` | Most GPUs running at once |
| `LEADROLE_WARM_GPUS` | `0` | Keep this many GPUs always on: no cold start, but you pay while idle |

After the GPUs have been idle, the first request waits for one to start, which can take a minute.
Uploaded videos and renders are deleted after 72 hours.

## Run on your own computer

Needs Python 3.10+ and `ffmpeg` on your PATH. With an NVIDIA GPU, install `onnxruntime-gpu` instead
of `onnxruntime`. On the CPU it works, but expect roughly 1 to 5 seconds per frame depending on the tier.

```bash
pip install numpy cython
pip install --no-build-isolation insightface==0.7.3   # Windows: install "C++ Build Tools" first
pip install -r requirements.txt
python run_local.py          # downloads the models (~1.2 GB) on first run
```

Then open http://127.0.0.1:8000.

## Project layout

```
modal_app.py          Modal deployment: web container, GPU workers, render orchestrator
run_local.py          everything in one process on your machine
faceswap/
  config.py           quality tiers, thresholds, model URLs
  engine.py           detection, identity matching, tracking, swap, restoration, blending
  media.py            ffmpeg: probe, frame-accurate streaming, encode, stitch + audio
  pipeline.py         studio operations: check photos, find people, preview, render chunks
  server.py           HTTP API and the web UI
  local.py            in-process backend
  static/             the web UI (no build step)
```

## Licences and responsible use

* **Face swap and recognition models.** `inswapper_128` and the `buffalo_l` detector/recogniser
  come from InsightFace, which releases them for **non-commercial research use**. Running ads is
  commercial use, so get a commercial licence from InsightFace or swap in a model you're licensed
  to use. The swap step is one function, `FaceEngine._swap_crop` in `faceswap/engine.py`.
* GFPGAN face restoration is Apache-2.0. The BiSeNet face parser is MIT.
* Only render faces you have permission to use. The app asks you to confirm this for every render.
* Most ad platforms require you to disclose realistic AI-altered people in ads. Check the policy
  of every platform you run the ad on.
