# FaceSwap Studio

Swap a face into ad videos on your Mac. Upload a video and a photo, mark the parts of
the video to change, choose whose face gets replaced, and download the finished MP4.

Everything runs on your Mac. The app is a web page served only to your own machine at
`http://127.0.0.1:7860`. No other computer or phone can open it, and your videos never
leave your Mac.

## What you need

- An Apple Silicon Mac (M1 or newer) with macOS 14 Sonoma or newer
- [Homebrew](https://brew.sh)
- About 3 GB of free disk space, plus room for your videos

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
3. downloads the face models (about 850 MB) and checks their fingerprints
4. measures whether your Mac's GPU or CPU is faster and uses the faster one

It's safe to run again; it skips anything already done.

## Start it

```bash
./run.sh
```

Or double-click **FaceSwap Studio.command** in Finder. Your browser opens the app. To
stop it, press Ctrl+C in Terminal (or close the Terminal window).

## How to use it

1. **Drop in your ad video.** Phone videos are fixed up automatically. That includes
   rotation, variable frame rate (which would make the audio drift) and HDR color. The
   app tells you when it changed something.
2. **Drop in the face photo.** Use a clear, front-facing, well-lit photo. If the photo
   has more than one face, click the one to use.
3. **Mark the segments to swap.** Only these frames go through the face models, which
   saves a lot of time. Pause where a swap should start and press **I**. Move to where
   it should end and press **O**. Repeat for each part. You can also type exact times
   into the segment list.
4. **Choose who gets swapped**, separately for each segment:
   - **One person:** only the person you pick. Pause on a frame that shows them, click
     **Pick person**, then click their face. The app follows them even when they turn
     their head. The **Match** slider sets how strict it is: lower catches them at more
     angles, higher avoids swapping someone who looks similar.
   - **Everyone:** every face in the frame.
   - **Whoever's on screen:** only the main face, meaning the biggest one. Use this
     when different people take turns on camera. It won't flip back and forth when two
     faces are a similar size.
5. **Preview this frame** swaps just the current frame and shows a before/after slider.
   Check the result before you render.
6. **Render.**
   - **Full video** keeps the same length and audio with only the segments swapped.
   - **Clips only** outputs just the segments, joined together.

   **Max quality** gives a slower, higher-quality encode. **Face detection: Fast** is
   quicker and works when faces are big in the frame.
7. **Finished videos** appear at the bottom with **Download** and **Reveal in Finder**.
   They're saved in `data/outputs/`.

Keyboard shortcuts:

| Key | Action |
| :-- | :-- |
| Space | Play / pause |
| `,` `.` or ← → | Step one frame (hold Shift for 10) |
| I / O | Set the segment start / end |
| Esc | Cancel picking a person or a half-made segment |

## Speed

A frame that isn't in a segment costs almost nothing. A frame in a segment costs one face
detection plus one swap per face that gets replaced. The setup script prints the speed
your Mac gets. To see it again, run `.venv/bin/python -m app.cli benchmark`.

Things that make a render faster: shorter segments, **Fast** face detection for close-ups,
and **One person** or **Whoever's on screen** instead of **Everyone** when there's a crowd.
4K footage is slower than 1080p.

## Troubleshooting

| What you see | What to do |
| :-- | :-- |
| "No face found in your photo" | Use a clear, front-facing photo without sunglasses or heavy shadow. |
| One person mode swaps the wrong person | Raise **Match**, or pick the person again from a clearer, front-on frame. |
| The swap drops out when they turn their head | Lower **Match** a little. |
| Whoever's on screen picks the wrong face | Split that part into its own segment and use **One person**. |
| The face looks soft in big close-ups | The swap model works at 128×128 pixels, so very large faces lose some detail. |
| Renders are slow | Check that the status at the top says "swapping on Apple GPU", then run `./scripts/setup.sh` again to re-measure. Mark shorter segments. |
| "Port 7860 is busy" | It's already running. Open http://127.0.0.1:7860. |
| "Face models missing" | Run `./scripts/setup.sh`. |
| The colors of an iPhone HDR video look flat | Export or record the clip in SDR (turn off HDR video) for the best color. |

If a render fails, the details are in `data/projects/<project>/last_render.log`.

## Licensing and consent (read before running paid ads)

- The face models, `inswapper_128` and `buffalo_l`, are published by InsightFace for
  **non-commercial research use**. Using them to make paid or commercial ads needs a
  commercial license from InsightFace, or a swap model whose license allows commercial use.
- Get written permission from the person whose face you put in and from the people in
  the footage.
- Ad platforms (Meta, Google, TikTok and others) have rules about disclosing AI-altered
  media. Check the current policy for each platform you advertise on.

## For developers

```
app/        server.py (web API), pipeline.py (render), faces.py (models), targets.py
            (who gets swapped), segments.py (time ranges), video_io.py (ffmpeg), cli.py
web/        the page: index.html, app.js, styles.css (no build step, no CDN)
tests/      pytest suite; e2e/ui_flow.cjs drives the UI in a browser
scripts/    setup.sh
```

- **Tests.** Run `.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest`.
- **Tests with the real models.** Put `test_video.mp4` (two people) and `test_face.jpg`
  in `samples/`. `tests/test_real_render.py` then runs on them.
- **Browser test.** Start the app with `./run.sh --no-browser`, then run
  `node tests/e2e/ui_flow.cjs`. It needs `npm install playwright`. Screenshots go to
  `tests/artifacts/`.
- **Command line.** For example:
  `.venv/bin/python -m app.cli render --video ad.mp4 --face me.jpg --segment 2-5.5:everyone --segment 9-12:main --out out.mp4`.
  Add `--clips` for clips only. For `:one` segments, add `--ref-time 14.5 --ref-face 0`.
- **Face engine.** `app/faces.py` re-implements the parts of InsightFace the app uses,
  so nothing has to be compiled. Paste-back only touches the area around each face.
  `tests/test_engine_parity.py` checks the output against the real insightface package
  when it's installed.
- **Network.** The server binds to `127.0.0.1` only, and that's hardcoded in
  `app/config.py`. It also rejects requests whose Host header isn't `127.0.0.1` or
  `localhost`, which blocks DNS rebinding. It requires a custom header on every write,
  so other websites open in your browser can't drive it.
