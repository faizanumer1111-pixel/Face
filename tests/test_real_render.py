"""End-to-end checks with the real face models on real footage.

Put a short clip with at least two people in samples/test_video.mp4 (or .mov) and a
face photo in samples/test_face.jpg (or .png), then run:
    pytest tests/test_real_render.py -s
These are skipped when the samples or models aren't there.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import MODELS_DIR
from app.faces import FaceEngine, missing_models
from app.pipeline import RenderRequest, render
from app.segments import Segment
from app.video_io import FrameReader, prepare_source, probe, read_frame

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def _find(stem, exts):
    for ext in exts:
        p = SAMPLES / f"{stem}{ext}"
        if p.exists():
            return p
    return None


VIDEO = _find("test_video", (".mp4", ".mov", ".m4v"))
FACE = _find("test_face", (".jpg", ".jpeg", ".png", ".heic"))
pytestmark = pytest.mark.skipif(VIDEO is None or FACE is None or bool(missing_models(MODELS_DIR)),
                                reason="needs samples/test_video.* + samples/test_face.* and downloaded models")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def frames_of(path, info, start=0, count=None):
    with FrameReader(path, info, start, count) as r:
        return [f.copy() for f in r]


@pytest.fixture(scope="module")
def engine():
    return FaceEngine(MODELS_DIR)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("real")
    info, _ = prepare_source(VIDEO, d / "work.mp4")
    return d, d / "work.mp4", info


def region(img, bbox, pad=0.1):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return img[int(max(0, y1 + h * pad)):int(y2 - h * pad), int(max(0, x1 + w * pad)):int(x2 - w * pad)]


def test_cli_two_second_segment(work):
    d, _, info = work
    fps = float(info.fps)
    start_s = round(1.0, 3)
    end_s = round(start_s + 2.0, 3)
    out = d / "cli_full.mp4"
    proc = subprocess.run([sys.executable, "-m", "app.cli", "render", "--video", str(VIDEO), "--face", str(FACE),
                           "--segment", f"{start_s}-{end_s}:everyone", "--out", str(out), "--json"],
                          cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout[-1500:])
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    seg_frames = round(end_s * fps) - round(start_s * fps)
    assert result["frames_swapped"] == seg_frames  # only segment frames went to the models

    src_info = probe(VIDEO)
    oinfo = probe(out)
    assert oinfo.total_frames == src_info.total_frames
    assert oinfo.fps == src_info.fps
    assert abs(oinfo.duration - src_info.duration) < 0.1
    assert oinfo.has_audio == src_info.has_audio

    s0, s1 = round(start_s * fps), round(end_s * fps)
    src = frames_of(d / "work.mp4", info)
    dst = frames_of(out, oinfo)
    outside = [psnr(src[i], dst[i]) for i in range(info.total_frames) if not s0 <= i < s1]
    inside = [psnr(src[i], dst[i]) for i in range(s0, s1)]
    print(f"outside segment: min {min(outside):.1f} dB, mean {np.mean(outside):.1f} dB; "
          f"inside segment: mean {np.mean(inside):.1f} dB")
    assert min(outside) > 40, "frames outside the segment changed"
    assert max(inside) < min(outside), "frames inside the segment weren't swapped"


def test_clips_only_duration(work, engine):
    d, path, info = work
    src_emb = _source_embedding(engine)
    segs = [Segment(10, 30, "main"), Segment(60, 70, "main")]
    out = d / "clips.mp4"
    stats = render(RenderRequest(path, info, src_emb, segs, {}, out, "clips", det_size=320), engine)
    oinfo = probe(out)
    assert stats.frames_swapped == 30
    assert oinfo.total_frames == 30
    assert abs(oinfo.duration - 30 / float(info.fps)) < 0.06


def _source_embedding(engine):
    import cv2

    img = cv2.imread(str(FACE))
    faces = engine.detect(img, 640)
    assert faces, "no face in samples/test_face"
    return engine.embed(img, max(faces, key=lambda f: f.area))


def test_one_person_swaps_only_them(work, engine):
    d, path, info = work
    frame = read_frame(path, info, 20)
    faces = sorted(engine.detect(frame, 640), key=lambda f: f.area, reverse=True)
    if len(faces) < 2:
        pytest.skip("sample needs two people visible at frame 20")
    target, other = faces[0], faces[1]
    ref = engine.embed(frame, target)
    out = d / "one.mp4"
    segs = [Segment(20, 32, "one", ref="r", threshold=0.35)]
    stats = render(RenderRequest(path, info, _source_embedding(engine), segs, {"r": ref}, out, "full"), engine)
    assert stats.faces_swapped == 12  # one face per frame
    oinfo = probe(out)
    src = frames_of(path, info, 20, 12)
    dst = frames_of(out, oinfo, 20, 12)
    for a, b in zip(src, dst):
        assert psnr(region(a, target.bbox), region(b, target.bbox)) < 30  # picked person changed
        assert psnr(region(a, other.bbox), region(b, other.bbox)) > 38  # everyone else untouched


def test_main_face_swaps_the_biggest(work, engine):
    d, path, info = work
    frame = read_frame(path, info, 40)
    faces = sorted(engine.detect(frame, 640), key=lambda f: f.area, reverse=True)
    if len(faces) < 2:
        pytest.skip("sample needs two people visible at frame 40")
    out = d / "main.mp4"
    stats = render(RenderRequest(path, info, _source_embedding(engine), [Segment(40, 46, "main")], {}, out, "full"),
                   engine)
    assert stats.faces_swapped == 6
    a = read_frame(path, info, 40)
    b = read_frame(out, probe(out), 40)
    assert psnr(region(a, faces[0].bbox), region(b, faces[0].bbox)) < 30
    assert psnr(region(a, faces[1].bbox), region(b, faces[1].bbox)) > 38
