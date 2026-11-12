import subprocess
from fractions import Fraction

import numpy as np
import pytest

from app.video_io import FrameReader, FrameWriter, choose_rate, prepare_source, probe, read_frame
from tests.conftest import make_video


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def all_frames(path, info):
    with FrameReader(path, info) as r:
        return [f.copy() for f in r]


def test_prepare_h264_is_remuxed_without_notes(small_video, tmp_path):
    info, notes = prepare_source(small_video, tmp_path / "work.mp4")
    assert notes == []
    assert (info.width, info.height, info.fps, info.total_frames) == (320, 240, Fraction(25), 75)
    assert info.has_audio and info.audio_codec == "aac"


def test_vfr_becomes_constant_rate(tmp_path):
    a = make_video(tmp_path / "a.mp4", seconds=1, fps="25", audio=False)
    b = make_video(tmp_path / "b.mp4", seconds=1, fps="50", audio=False)
    lst = tmp_path / "list.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    vfr = tmp_path / "vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(vfr)], check=True)
    info, notes = prepare_source(vfr, tmp_path / "work.mp4")
    assert any("constant" in n for n in notes)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,avg_frame_rate", "-of", "csv=p=0", str(tmp_path / "work.mp4")],
                         capture_output=True, text=True).stdout.strip().split(",")
    assert Fraction(out[0]) == info.fps
    assert abs(info.total_frames - round(2.0 * float(info.fps))) <= 1


def test_rotation_is_applied(small_video, tmp_path):
    rotated = tmp_path / "rot.mp4"
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", str(small_video),
                           "-c", "copy", str(rotated)], capture_output=True)
    if proc.returncode != 0:
        pytest.skip("this ffmpeg can't write rotation metadata")
    info, notes = prepare_source(rotated, tmp_path / "work.mp4")
    assert (info.width, info.height) == (240, 320)
    assert any("rotation" in n for n in notes)


def test_odd_size_and_444_are_converted(tmp_path):
    src = make_video(tmp_path / "odd.mp4", size="321x241", audio=False, extra=["-pix_fmt", "yuv444p"])
    info, _ = prepare_source(src, tmp_path / "work.mp4")
    assert (info.width, info.height) == (320, 240)
    assert info.pix_fmt == "yuv420p"


def test_choose_rate():
    assert choose_rate(Fraction(30), Fraction(2987, 100)) == Fraction(30)
    assert choose_rate(Fraction(600), Fraction(2997, 100)) == Fraction(30000, 1001)


def test_seeking_is_frame_accurate(small_video, tmp_path):
    info, _ = prepare_source(small_video, tmp_path / "work.mp4")
    work = tmp_path / "work.mp4"
    frames = all_frames(work, info)
    assert len(frames) == info.total_frames
    for start in (1, 10, 13, 50, 74):
        with FrameReader(work, info, start=start, count=min(5, info.total_frames - start)) as r:
            for k, f in enumerate(r):
                assert np.array_equal(f, frames[start + k]), f"frame {start + k} differs after seeking"
    assert np.array_equal(read_frame(work, info, 37), frames[37])


def test_encode_round_trip_keeps_colour_and_audio(small_video, tmp_path):
    info, _ = prepare_source(small_video, tmp_path / "work.mp4")
    work = tmp_path / "work.mp4"
    frames = all_frames(work, info)
    out = tmp_path / "out.mp4"
    w = FrameWriter(out, info, work, "full", False, tmp_path / "enc.log")
    for f in frames:
        w.write(f)
    w.finish()
    info2 = probe(out)
    assert (info2.total_frames, info2.fps, info2.width) == (info.total_frames, info.fps, info.width)
    assert info2.has_audio
    again = all_frames(out, info2)
    scores = [psnr(a, b) for a, b in zip(frames, again)]
    # Only compression loss is left (the strict >40 dB per-frame check runs on real footage).
    assert np.mean(scores) > 40 and min(scores) > 37
