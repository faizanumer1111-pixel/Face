import json
import math
import subprocess

import cv2
import numpy as np
import pytest

from app.faces import Face
from app.pipeline import RenderRequest, render
from app.segments import Segment
from app.video_io import prepare_source, probe
from app.vr import EquirectView, Layout, detect_format, eyes, match_between_eyes, view_for_face
from tests.conftest import make_video
from tests.fakes import LEFT_ID


def face(x, y, size):
    return Face(bbox=np.array([x, y, x + size, y + size], np.float32), kps=np.zeros((5, 2)), score=0.9)


def test_detect_format_from_tags_and_names():
    sbs180 = {"side_data_list": [
        {"side_data_type": "Spherical Mapping", "projection": "equirectangular", "bound_left": 1073741824,
         "bound_right": 1073741824},
        {"side_data_type": "Stereo 3D", "type": "side by side"}]}
    assert detect_format(sbs180, "clip.mp4", 5760, 2880)[0] == "vr180_sbs"
    assert detect_format({}, "my_ad_180_LR.mp4", 5760, 2880)[0] == "vr180_sbs"
    assert detect_format({}, "beach_360_TB.mp4", 4096, 4096)[0] == "vr360_tb"
    assert detect_format({}, "beach_360.mp4", 3840, 1920)[0] == "vr360_mono"
    assert detect_format({}, "movie_3dh.mp4", 3840, 1080)[0] == "flat_sbs"
    assert detect_format({}, "summer sale.mp4", 1920, 1080) == ("flat", None)
    assert detect_format({}, "ad_1080p.mp4", 1920, 1080)[0] == "flat"


def test_eyes_are_views_into_the_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    left, right = eyes(frame, Layout.of("vr180_sbs"))
    right[:] = 7
    assert frame[:, 100:].min() == 7 and frame[:, :100].max() == 0
    top, bottom = eyes(np.zeros((200, 100, 3), np.uint8), Layout.of("vr360_tb"))
    assert top.shape == bottom.shape == (100, 100, 3)


def test_match_between_eyes_pairs_the_same_person():
    left = [face(100, 100, 80), face(400, 120, 60)]
    right = [face(380, 118, 62), face(90, 100, 82)]  # shifted a little, listed in another order
    pairs = match_between_eyes(left, right)
    assert pairs[0] is right[1] and pairs[1] is right[0]
    assert match_between_eyes([face(100, 100, 80)], [face(900, 600, 80)]) == {}


@pytest.mark.parametrize("projection", ["equirect180", "equirect360"])
def test_view_round_trip_changes_only_the_face(projection):
    rng = np.random.default_rng(0)
    eye = cv2.GaussianBlur(rng.integers(0, 255, (720, 720 if projection == "equirect180" else 1440, 3),
                                        dtype=np.uint8), (9, 9), 3)
    original = eye.copy()
    v = view_for_face(eye.shape, projection, np.array([500, 200, 560, 260], np.float32))
    img = v.render(eye)
    edited = img.copy()
    c = v.size // 2
    cv2.circle(edited, (c, c), v.size // 8, (0, 255, 0), -1)
    v.paste(eye, img, edited)
    changed = np.abs(eye.astype(int) - original.astype(int)).max(axis=2) > 0
    ys, xs = np.nonzero(changed)
    # the change landed around the face, nowhere else
    assert 470 < xs.mean() < 590 and 170 < ys.mean() < 290
    assert changed.sum() < 0.05 * changed.size
    # and looking through the same view again shows the green disc
    again = v.render(eye)
    assert again[c, c, 1] > 200 and again[c, c, 2] < 60


def test_view_points_at_the_face():
    v = view_for_face((1000, 1000), "equirect180", np.array([700, 300, 760, 360], np.float32))
    assert math.degrees(v.yaw) == pytest.approx(((730.5 / 1000) * 180) - 90, abs=0.5)
    assert v.pitch > 0  # above the horizon


def _vr_clip(tmp_path, name="ad_180_LR.mp4"):
    src = make_video(tmp_path / "flat.mp4", size="320x240", seconds=1.0)
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    "[0:v]split[a][b];[a]v360=input=flat:output=hequirect:ih_fov=100:iv_fov=75:w=480:h=480[l];"
                    "[b]v360=input=flat:output=hequirect:ih_fov=100:iv_fov=75:w=480:h=480:yaw=1[r];"
                    "[l][r]hstack,format=yuv420p[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264",
                    "-c:a", "aac", str(out)], check=True)
    return out


def test_vr_render_swaps_both_eyes_and_tags_the_file(tmp_path, fake_engine):
    clip = _vr_clip(tmp_path)
    info, notes = prepare_source(clip, tmp_path / "work.mp4")
    assert info.vr_format == "vr180_sbs" and any("VR 180" in n for n in notes)
    out = tmp_path / "out_180_LR.mp4"
    req = RenderRequest(tmp_path / "work.mp4", info, {"f": LEFT_ID}, "f", [Segment(5, 10, "everyone")], {},
                        out, "full", vr_format="vr180_sbs")
    stats = render(req, fake_engine)
    assert stats.frames_swapped == 5
    assert stats.faces_swapped == 5 * 4  # two fake faces in each eye

    data = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                                      str(out)], capture_output=True, text=True).stdout)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert v["codec_name"] == "hevc"
    kinds = {sd.get("side_data_type") for sd in v.get("side_data_list", [])}
    assert "Spherical Mapping" in kinds and "Stereo 3D" in kinds
    stereo = next(sd for sd in v["side_data_list"] if sd["side_data_type"] == "Stereo 3D")
    assert stereo["type"] == "side by side"
    # and the tags make the app recognise its own output as VR again
    again, _ = prepare_source(out, tmp_path / "again.mp4")
    assert again.vr_format == "vr180_sbs"
    assert probe(out).total_frames == info.total_frames
