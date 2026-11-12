import threading

import numpy as np
import pytest

from app.pipeline import Cancelled, RenderRequest, render
from app.segments import Segment
from app.video_io import FrameReader, prepare_source, probe
from tests.fakes import LEFT_ID


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def frames_of(path, info):
    with FrameReader(path, info) as r:
        return [f.copy() for f in r]


@pytest.fixture
def work(small_video, tmp_path):
    info, _ = prepare_source(small_video, tmp_path / "work.mp4")
    return tmp_path / "work.mp4", info


def boxes(shape):
    h, w = shape[:2]
    left = (slice(int(h * .2) + 2, int(h * .8) - 2), slice(int(w * .1) + 2, int(w * .4) - 2))
    right = (slice(int(h * .35) + 2, int(h * .65) - 2), slice(int(w * .65) + 2, int(w * .8) - 2))
    return left, right


def test_full_render_swaps_only_inside_segments(work, tmp_path, fake_engine):
    path, info = work
    segs = [Segment(10, 20, "everyone"), Segment(40, 50, "main"), Segment(60, 65, "one", ref="me")]
    out = tmp_path / "out.mp4"
    req = RenderRequest(path, info, LEFT_ID, segs, {"me": LEFT_ID}, out, "full")
    stats = render(req, fake_engine)

    # only segment frames went to the face models
    assert fake_engine.detect_calls == stats.frames_swapped == 25
    assert stats.faces_swapped == 10 * 2 + 10 * 1 + 5 * 1

    oinfo = probe(out)
    assert (oinfo.total_frames, oinfo.fps, oinfo.width, oinfo.height) == (75, info.fps, 320, 240)
    assert abs(oinfo.duration - info.duration) < 0.05
    assert oinfo.has_audio

    src, dst = frames_of(path, info), frames_of(out, oinfo)
    left, right = boxes(src[0].shape)
    in_seg = set(range(10, 20)) | set(range(40, 50)) | set(range(60, 65))
    outside = [psnr(src[i], dst[i]) for i in range(75) if i not in in_seg]
    assert np.mean(outside) > 40 and min(outside) > 37, "frames outside the segments changed"
    for i in range(10, 20):  # everyone: both faces changed
        assert psnr(src[i][left], dst[i][left]) < 20 and psnr(src[i][right], dst[i][right]) < 20
    for i in list(range(40, 50)) + list(range(60, 65)):  # main / one: only the left face
        assert psnr(src[i][left], dst[i][left]) < 20
        assert psnr(src[i][right], dst[i][right]) > 35


def test_clips_render_is_just_the_segments(work, tmp_path, fake_engine):
    path, info = work
    segs = [Segment(10, 20, "everyone"), Segment(50, 62, "everyone")]
    out = tmp_path / "clips.mp4"
    render(RenderRequest(path, info, LEFT_ID, segs, {}, out, "clips"), fake_engine)
    oinfo = probe(out)
    assert oinfo.total_frames == 22
    assert abs(oinfo.duration - 22 / 25) < 0.05
    assert oinfo.has_audio
    dst = frames_of(out, oinfo)
    src = frames_of(path, info)
    left, _ = boxes(src[0].shape)
    # first clip frame is source frame 10, first frame of the 2nd clip is source frame 50 (swapped)
    expected = src[10].copy()
    expected[left] = 255 - expected[left]
    assert psnr(dst[0][left], expected[left]) > 30
    expected = src[50].copy()
    expected[left] = 255 - expected[left]
    assert psnr(dst[10][left], expected[left]) > 30


def test_cancel_leaves_no_file(work, tmp_path, fake_engine):
    path, info = work
    out = tmp_path / "cancelled.mp4"
    cancel = threading.Event()
    calls = {"n": 0}
    orig = fake_engine.detect

    def slow_detect(img, det_size=640):
        calls["n"] += 1
        if calls["n"] == 5:
            cancel.set()
        return orig(img, det_size)

    fake_engine.detect = slow_detect
    with pytest.raises(Cancelled):
        render(RenderRequest(path, info, LEFT_ID, [Segment(0, 75, "everyone")], {}, out, "full"), fake_engine,
               cancel=cancel)
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial.mp4"))
