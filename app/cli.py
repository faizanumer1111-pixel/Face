"""Command line: render without the web page, and benchmark the models.

    python -m app.cli render --video ad.mp4 --face me.jpg \
        --segment 2.0-5.5:everyone --segment 9-12:main --out out.mp4
    python -m app.cli render ... --segment 14-16:one --ref-time 14.5 --ref-face 0
    python -m app.cli benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import config
from .faces import FaceEngine, load_provider_settings
from .pipeline import RenderRequest, render
from .segments import MODES, SegmentError, from_json, normalize, parse_time, stats
from .video_io import VideoError, load_image, prepare_source, read_frame


def parse_segment_spec(spec: str) -> dict:
    """"START-END[:MODE]" -> segment dict (times stay as text, parsed later)."""
    body, mode = spec, "everyone"
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        if tail in MODES:
            body, mode = head, tail
    if "-" not in body:
        raise SegmentError(f'Segment "{spec}" should look like START-END:MODE, e.g. 2.0-5.5:everyone')
    start, end = body.split("-", 1)
    return {"start": start, "end": end, "mode": mode}


def _largest(faces):
    return max(faces, key=lambda f: f.area)


def cmd_render(args) -> int:
    engine = FaceEngine(config.MODELS_DIR, load_provider_settings(config.DATA_DIR))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        work = td / "work.mp4"
        info, notes = prepare_source(Path(args.video), work)
        for n in notes:
            print("Note:", n)
        photo = load_image(Path(args.face), td)
        faces = engine.detect(photo, 640)
        if not faces:
            print("No face found in your photo. Use a clear, front-facing shot.", file=sys.stderr)
            return 2
        source = _largest(faces)
        src_emb = engine.embed(photo, source)

        items = [parse_segment_spec(s) for s in args.segment]
        refs: dict[str, np.ndarray] = {}
        if any(it["mode"] == "one" for it in items):
            if args.ref_time is None:
                print("One person mode needs --ref-time (a moment showing that person) and --ref-face.", file=sys.stderr)
                return 2
            ref_frame = parse_time(args.ref_time, info.fps)
            frame = read_frame(work, info, ref_frame)
            ref_faces = sorted(engine.detect(frame, args.det_size), key=lambda f: f.bbox[0])
            if not ref_faces or args.ref_face >= len(ref_faces):
                print(f"Only {len(ref_faces)} face(s) at that time; --ref-face counts from 0, left to right.", file=sys.stderr)
                return 2
            refs["cli"] = engine.embed(frame, ref_faces[args.ref_face])
            for it in items:
                if it["mode"] == "one":
                    it["ref"] = "cli"
                    it["threshold"] = args.threshold
        segs = normalize(from_json(items, info.fps), info.total_frames)
        st = stats(segs, info.total_frames)
        print(f"Processing {st['frames_to_process']} of {st['total_frames']} frames ({st['percent_skipped']}% skipped)")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        req = RenderRequest(video_path=work, info=info, source_embedding=src_emb, segments=segs, references=refs,
                            output_path=out, output_mode="clips" if args.clips else "full",
                            det_size=args.det_size, max_quality=args.max_quality)

        def progress(s):
            eta = f"{s.eta_seconds:.0f}s" if s.eta_seconds is not None else "?"
            print(f"\r  swapped {s.frames_swapped}/{s.swap_total} · written {s.frames_written}/{s.frames_total}"
                  f" · {s.swap_fps:.1f} fps · ETA {eta}   ", end="", flush=True)

        t0 = time.time()
        result = render(req, engine, on_progress=progress, log_path=td / "encode.log")
        print()
        print(f"Done in {time.time() - t0:.1f}s -> {out}")
        print(f"Frames sent to the face models: {result.frames_swapped} (segment frames: {st['frames_to_process']})")
        print(f"Faces swapped: {result.faces_swapped}; frames with nobody to swap: {result.frames_without_target}")
        if args.json:
            print(json.dumps(result.to_json()))
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import run_benchmark

    report = run_benchmark(config.MODELS_DIR, config.DATA_DIR, runs=args.runs)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    names = {"det": "Face detection", "rec": "Face matching", "swap": "Face swap"}
    labels = {"coreml": "Mac GPU (CoreML)", "cuda": "NVIDIA GPU", "cpu": "CPU"}
    for key, m in report["models"].items():
        parts = []
        for choice, r in m["results"].items():
            if "ms_per_run" in r and r.get("correct", True):
                parts.append(f"{labels[choice]} {r['ms_per_run']:.0f} ms")
            elif "ms_per_run" in r:
                parts.append(f"{labels[choice]} gave wrong results ({r.get('check', '')})")
            else:
                parts.append(f"{labels[choice]} not usable ({r.get('error', 'failed')})")
        print(f"  {names[key]:15s} -> using {labels[m['picked']]:17s} [{'; '.join(parts)}]")
    fps = report.get("estimated_fps_one_face")
    if fps:
        print(f"  Estimated speed with one face per frame: about {fps} frames per second")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.cli", description="FaceSwap Studio command line")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="swap faces in parts of a video")
    r.add_argument("--video", required=True)
    r.add_argument("--face", required=True, help="photo of the face to put in")
    r.add_argument("--segment", action="append", required=True,
                   help="START-END:MODE, e.g. 2.0-5.5:everyone, 9-12:main, 14-16:one (repeatable)")
    r.add_argument("--out", required=True)
    r.add_argument("--clips", action="store_true", help="output only the segments, joined")
    r.add_argument("--max-quality", action="store_true", help="slower, higher quality encode")
    r.add_argument("--det-size", type=int, default=640, choices=[320, 480, 640])
    r.add_argument("--ref-time", help="for :one segments, a time showing the person to swap")
    r.add_argument("--ref-face", type=int, default=0, help="which face at --ref-time, 0 = leftmost")
    r.add_argument("--threshold", type=float, default=0.35, help="match strictness for :one segments")
    r.add_argument("--json", action="store_true", help="print render stats as JSON")
    r.set_defaults(func=cmd_render)

    b = sub.add_parser("benchmark", help="time each model on CoreML/CPU and save the fastest")
    b.add_argument("--runs", type=int, default=20)
    b.add_argument("--json", action="store_true", help="print the full report as JSON")
    b.set_defaults(func=cmd_benchmark)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (SegmentError, VideoError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
