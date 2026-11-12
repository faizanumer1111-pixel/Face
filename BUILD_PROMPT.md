I want you to build **FaceSwap Studio**: an app that runs entirely on my Mac and swaps faces in ad videos. I upload a video and a photo of a face, mark the parts of the video I want changed, choose whose face gets replaced, and get back a rendered MP4. It has a web interface served only to my own machine.

## Context

- Hardware: an Apple Silicon Mac. There's no NVIDIA GPU, so use ONNX Runtime with the `CoreMLExecutionProvider` and fall back to `CPUExecutionProvider`.
- If `faceswap modal.py` is in this folder, read it first. It's my earlier cloud version, and its pipeline already works: InsightFace `buffalo_l` for detection and recognition, `inswapper_128.onnx` for the swap, and ffmpeg to re-encode and copy the original audio back in. Port that pipeline. Leave out everything Modal-specific.
- Everything runs offline after setup. The only network access is the one-time download of models and packages.

## What it does

1. **Inputs.** I upload a video (MP4/MOV, including phone footage) and a source photo. Detect the face in the photo and show me the crop. If the photo has no face, tell me clearly. If it has several, let me click the one to use.
2. **Segments (this is how I save GPU time).** I mark one or more time ranges on a timeline. Only frames inside those ranges get face detection and swapping. Frames outside them are never sent to any model.
   - Timeline scrubber with the segments drawn as colored bars on it. `I` sets the in point and `O` sets the out point. `,` and `.` step one frame. The segment list shows editable start and end timecodes and a delete button.
   - Merge overlapping segments. Snap segment edges to exact frame numbers so the boundaries are frame-accurate.
   - Show the savings before I render, for example "Processing 240 of 1,800 frames (87% skipped)".
3. **Who gets swapped.** Each segment has its own target mode, and a new segment starts with the last mode I used:
   - **One person.** I pause on a frame and the app draws boxes around every detected face. I click one, and it becomes the reference. In each frame, swap only faces whose embedding has cosine similarity to the reference above a threshold. The threshold has a slider (default 0.35). Add simple IoU tracking so the swap doesn't flicker off for a few frames when the similarity dips (profile views, motion blur).
   - **Everyone.** Swap every detected face.
   - **Whoever's on screen.** Swap only the main face in each frame: the largest one, with ties broken by which is closest to the center. This is for videos where different people take turns on camera. Use hysteresis: switch the main face only after another face has been clearly bigger (about 1.3×) for several frames in a row. Otherwise two faces of similar size make the swap jump back and forth.
4. **Preview.** A "Preview this frame" button swaps only the current frame with the current settings and shows a before/after comparison, so I can check the result before a full render.
5. **Render.** Two output choices:
   - **Full video:** the same length, resolution, frame rate, and audio as the original, with only the segments swapped.
   - **Clips only:** just the segments, joined together in order. This mode seeks straight to each segment and doesn't decode the rest of the video.

   Show a progress bar with frames done/total, current fps, and ETA. Include a Cancel button that actually stops the job. Finished renders go in an outputs list with Download and "Reveal in Finder" buttons.

## How to build it

- **Stack:** Python 3.11 in a `.venv`, FastAPI with uvicorn, `insightface`, `onnxruntime` (the standard macOS arm64 wheel includes CoreML), `opencv-python-headless`, and ffmpeg from Homebrew. The frontend is plain HTML, CSS, and JS served by FastAPI, with no build step and no CDN links. Every asset is served locally.
- **Local only.** Hardcode the bind host to `127.0.0.1`, never `0.0.0.0`, and don't make it configurable. Reject requests whose `Host` header isn't `127.0.0.1:<port>` or `localhost:<port>`, which blocks DNS rebinding. Use port 7860. Add a test that fails if the server can be reached on any interface other than loopback.
- **Speed:**
  - Load models once at startup.
  - Only load the InsightFace modules that are actually used. Use `allowed_modules=['detection','recognition']`, and in the Everyone and Whoever's-on-screen modes skip recognition on target faces entirely, because the swapper only needs their keypoints.
  - Make detection size a setting: 640 by default, 320 for speed.
  - Run one job at a time through a single worker queue, since there's one GPU.
- **Video I/O:**
  - Decode and encode with ffmpeg subprocess pipes (rawvideo bgr24). Don't use OpenCV for this, because ffmpeg handles rotation metadata on phone video and keeps timing exact.
  - Run ffprobe at upload time. If the video has a variable frame rate, convert it to constant frame rate at that point and tell me in the UI, so the audio doesn't drift out of sync.
  - Encode with `h264_videotoolbox` at high quality. Use `libx264 -crf 18` as the fallback, and also when I tick a "max quality" option. Copy the original audio across.
- **Storage:** Keep everything under `data/` in the project, with folders for uploads, projects, and outputs. Add `data/` to `.gitignore`. Save each project's segments and settings as JSON so I can reload a project later.
- **Commands:**
  - `scripts/setup.sh` checks for or installs Homebrew ffmpeg, creates the venv, installs the requirements, downloads the models into `models/`, and prints their sha256 hashes.
  - `./run.sh` starts the server and opens `http://127.0.0.1:7860`.
  - Also add a CLI, `python -m app.cli --video V --face F --segment 2.0-5.5:everyone --segment 9-12:main --out out.mp4`, so the pipeline can be tested without the UI.

Suggested layout (change it if you have a good reason): `app/` (server, pipeline, targets, segments, video_io, models, cli), `web/`, `tests/`, `scripts/`, `samples/`, `models/`, `data/`.

## Working process

**Phase 0. Check, then stop.** Read-only checks first: chip, macOS version, free disk, whether python3.11, Homebrew, and ffmpeg are installed, and whether `samples/` contains a test video with at least two people and a face photo. Report what you found and give me a short build plan. Then wait for my go-ahead before you install anything. If the samples are missing, ask me to add them. They're needed for the end-to-end checks.

After I approve, build in this order. Run the check at the end of each step and paste the actual output. Don't just say that it passed.

1. **Setup and models.** Check: `scripts/setup.sh` runs cleanly. Benchmark swap fps on 100 frames of the sample with CoreML versus CPU, report both numbers, and default to the faster one. If CoreML fails on some model ops, say which ones.
2. **Segment logic and target selection.** Check: pytest unit tests for timecode ↔ frame conversion, merging overlaps, frame-accurate edges, and all three target modes (use mocked face detections), including the hysteresis and the gap tolerance in tracking.
3. **Render pipeline and CLI.** Check on the sample with one 2-second segment:
   - ffprobe shows the output's frame count, fps, and duration match the input, and audio is present.
   - Frames outside the segment match the source at PSNR > 40 dB.
   - Frames inside the segment are different from the source.
   - Log how many frames went through the model, and confirm it equals the segment's frame count.
   - Run Clips-only mode and confirm the output duration equals the total length of the segments.
4. **Web UI.** Check: drive the full flow with Playwright (upload, mark two segments with different modes, pick a reference face, preview, render, download) and save screenshots to `tests/artifacts/`. Then run `lsof -iTCP -sTCP:LISTEN -P | grep 7860` and show that it lists only `127.0.0.1`.
5. **README.** Cover setup, running, the three modes in plain English, troubleshooting (no face found, wrong person swapped, slow renders), and the licensing note below.

Keep the code simple: no auth, accounts, Docker, or database. Error messages shown in the UI should tell me what to do next, for example "No face found in your photo — use a clear, front-facing shot."

**Licensing note for the README.** `inswapper_128` and `buffalo_l` are published by InsightFace for non-commercial research use. Put a clear notice in the README that commercial ad use needs a license from InsightFace, or a different model with a commercial license.
