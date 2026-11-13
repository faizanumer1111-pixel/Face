# FaceSwap Studio

Recast the faces in your ad videos, on your Mac. Drop in a video and some face photos,
mark the moments to change, drag a face onto each person, and render.

Everything runs on your Mac. The app is a web page served only to your own machine at
`http://127.0.0.1:7860`. No other computer or phone can open it, and your videos never
leave your Mac.

## What it does

- **Casting board.** Keep a library of faces and drag one onto any person in the video.
  Different people in the same scene can get different faces.
- **Swap only what you mark.** Mark segments with **I** and **O**. Frames outside them are
  copied straight through and never touch the face models, so renders stay fast.
- **Three ways to choose who's swapped**, set per segment:
  - **Picked people:** each person you cast.
  - **Everyone:** every face in frame.
  - **Whoever's on screen:** the main, biggest face.
- **Faces that look real:**
  - The new face is sharpened.
  - Hands, hair and objects stay in front of the face.
  - The face is matched to the scene's lighting.
  - Its position is smoothed between frames, so it doesn't shimmer.
- **VR and 3D:** 180° and 360°, side-by-side or top-bottom. Both eyes get the same swap.
  Faces away from the centre of the VR frame are straightened before swapping and curved
  back afterwards. The VR tags are kept so headsets and YouTube play the result as VR.

## What you need

- An Apple Silicon Mac (M1 or newer) with macOS 14 Sonoma or newer
- [Homebrew](https://brew.sh)
- About 4 GB of free disk space, plus room for your videos

## Set up (once)

In Terminal:

```bash
git clone https://github.com/faizanumer1111-pixel/Face.git
cd Face
./scripts/setup.sh
```

The setup script:

1. installs ffmpeg and Python 3.12 with Homebrew
2. creates the Python environment
3. downloads the face models (about 1.3 GB) and checks their fingerprints
4. measures whether your Mac's GPU or CPU is faster and uses the faster one

It's safe to run again; it skips anything already done.

## Start it

```bash
./run.sh
```

Or double-click **FaceSwap Studio.command** in Finder. Your browser opens the app. To
stop it, press Ctrl+C in Terminal (or close the Terminal window).

## How to use it

1. **Add faces.** Drop photos onto the **Faces** panel. Use clear, front-facing,
   well-lit photos. A group photo adds every face in it; delete the ones you don't need.
   Double-click a name to rename it. The **★ starred** face is used wherever you
   haven't picked one.
2. **Drop in your ad video.** Phone footage is fixed up automatically: rotation, variable
   frame rate (which would make the audio drift) and HDR colour. VR and 3D are detected
   from the file's tags or its name (for example `_180_LR` or `_360_TB`). If the app
   gets the format wrong, change it under **Video format**.
3. **Mark the segments to swap.** Pause where a swap should start and press **I**. Move to
   where it should end and press **O**. You can also type exact times into a segment.
4. **Cast.** Pause inside a segment and boxes appear on everyone's face. Drag a face from
   the Faces panel onto a person, or click a person and choose a face. Each casting
   applies to the whole segment, and the app follows that person even when they turn
   their head.
   - **Match** sets how strict the following is. Lower catches the person at more
     angles; higher avoids swapping someone who looks similar.
   - **Everyone** and **Whoever's on screen** segments use the face you choose in that
     segment's row.
5. **Preview this frame** swaps just the current frame and shows a before/after slider.
6. **Choose the look** in the **Look** panel:
   - **Sharper faces:** 0% to 100%. Higher is crisper but slower.
   - **Keep hands and hair in front of the face**
   - **Match the scene's lighting**
   - **Steady faces**

   All are on by default.
7. **Render.**
   - **Full video** keeps the same length and sound with only the segments swapped.
   - **Just the segments** joins the marked parts into one clip.

   **Max quality** gives a slower, higher-quality encode. Face detection can be
   **Fast** (big close-ups), **Standard** or **Thorough** (small or distant faces).
   You don't have to wait for a render to finish. Keep working, add new videos, and
   press **Add to queue** as often as you like, including on the same project with
   different faces (handy for A/B variants). Each render keeps the faces and settings it
   had when you queued it. The **Render queue** panel runs them one after another, top to
   bottom: use **Up** and **Down** to change the order, and **Remove** or **Cancel** to
   drop one. One at a time is on purpose: two renders at once would share the same GPU
   and both finish later.
8. **Rendered videos** appear at the bottom with **Download**, **Reveal in Finder** and
   **Delete**. They're saved in `data/outputs/`. VR files end with their format tag,
   such as `_180_LR`.

Keyboard shortcuts:

| Key | Action |
| :-- | :-- |
| Space | Play / pause |
| `,` `.` or ← → | Step one frame (hold Shift for 10) |
| I / O | Mark the segment start / end |
| Esc | Close the face chooser, or cancel a half-made segment |

## Speed

A frame that isn't in a segment costs almost nothing. A frame in a segment costs a face
detection plus, for each face replaced, a swap. That swap includes sharpening if it's
on, and sharpening roughly doubles the time per face. The setup script prints the speed
your Mac gets. To see it again, run `.venv/bin/python -m app.cli benchmark`.

VR is the heaviest case. Both eyes are swapped, VR is often 60 frames a second, and 8K
frames are about 15× the pixels of 1080p. Expect VR renders to take several times longer
per second of footage than a normal ad, so mark only the moments you need.

When a render finishes, the queue says where most of the time went, for example
"sharpening (72 ms a frame)". To go faster:

- **Quick drafts:** set **Sharper faces** to 0%. Sharpening is usually the slowest step,
  so this roughly halves render time. Turn it back up for the final version.
- **Big, close-up faces:** use **Fast** face detection.
- **Only need the swapped moments:** choose **Just the segments**. The rest of the video
  isn't processed at all.
- **Short segments:** mark only the moments that need a different face.

The app already does these on its own:

- It uses the Mac's built-in video engine to decode and encode footage.
- It makes a light preview copy of 4K and VR videos, so scrubbing stays smooth.
- It skips sharpening on small faces, where it adds nothing.

## Troubleshooting

| What you see | What to do |
| :-- | :-- |
| "No face found in your photo" | Use a clear, front-facing photo without sunglasses or heavy shadow. |
| No boxes appear on a paused frame | The person may be too small or turned away. Try **Thorough** face detection, or a nearby frame. |
| A picked person stops being swapped when they turn | Lower **Match** a little. |
| The wrong person gets someone's face | Raise **Match**, or cast them again from a clearer, front-on frame. |
| A hand or microphone looks smeared | Make sure **Keep hands and hair in front** is on. |
| The face looks plastic | Turn **Sharper faces** down. |
| The face is too bright or dark for the scene | Turn on **Match the scene's lighting**. |
| VR swap looks wrong | Check **Video format** matches the footage (180 vs 360, side-by-side vs top-bottom). |
| Renders are slow | Check the status at the top says "swapping on the Mac GPU", then run `./scripts/setup.sh` again to re-measure. Mark shorter segments, or turn sharpening down. |
| "Port 7860 is busy" | It's already running. Open http://127.0.0.1:7860. |
| "Face models missing" | Run `./scripts/setup.sh`. |

If a render fails, the details are in `data/projects/<project>/last_render.log`.

## Licensing and consent (read before running paid ads)

- The face models have their own licenses, and they're not the app's code:
  - **inswapper_128** and **buffalo_l** (InsightFace) are published for
    **non-commercial research use**. Paid or commercial ads need a commercial license
    from InsightFace, or a swap model whose license allows commercial use.
  - **GFPGAN 1.4** (sharpening) is released by Tencent ARC under Apache 2.0.
  - The **XSeg** occlusion model comes from the DeepFaceLab and FaceFusion projects.
    Check its terms before commercial use.
- Get written permission from the people whose faces you put in and from the people in
  the footage.
- Ad platforms (Meta, Google, TikTok and others) have rules about disclosing AI-altered
  media. Check the current policy for each platform you advertise on.
- The bundled fonts are open source; see `web/fonts/LICENSES.md`.

## For developers

```
app/        server.py (web API), pipeline.py (render), processor.py (one frame: normal/3D/VR),
            faces.py (models), targets.py (who gets swapped, smoothing), segments.py (time ranges),
            vr.py (VR formats, eyes, stretch fix, VR tags), video_io.py (ffmpeg), projects.py
            (projects + face library), cli.py
web/        the page: index.html, app.js, styles.css, fonts/ (no build step, no CDN)
tests/      pytest suite; e2e/ui_flow.cjs drives the UI in a browser
scripts/    setup.sh
```

- **Tests.** Run `.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest`.
- **Tests with the real models.** Put `test_video.mp4` (two people), `test_face.jpg` and
  `test_face2.jpg` in `samples/`. `tests/test_real_render.py` then runs on them.
- **Browser test.** Start the app with `./run.sh --no-browser`, then run
  `node tests/e2e/ui_flow.cjs [video] [face] [face2]`. It needs `npm install playwright`.
  Screenshots go to `tests/artifacts/`.
- **Command line.** For example:
  `.venv/bin/python -m app.cli render --video ad.mp4 --face me.jpg --segment 2-5.5:everyone --segment 9-12:main --out out.mp4`.
  - `--clips` outputs just the segments.
  - `--vr-format vr180_sbs` sets the VR format.
  - `--sharpen 0` turns sharpening off; `--no-occlusion`, `--no-light-match` and
    `--no-steady` turn the other improvements off.
  - For a single picked person, use a `:one` segment with `--ref-time 14.5 --ref-face 0`.
- **Face engine.** `app/faces.py` re-implements the parts of InsightFace the app uses, so
  nothing has to be compiled. `tests/test_engine_parity.py` checks the output against the
  real insightface package when it's installed.
- **Network.** The server binds to `127.0.0.1` only, and that's hardcoded in
  `app/config.py`. It also rejects requests whose Host header isn't `127.0.0.1` or
  `localhost`, which blocks DNS rebinding. It requires a custom header on every write, so
  other websites open in your browser can't drive it.
