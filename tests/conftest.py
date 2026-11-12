import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def make_video(path: Path, seconds: float = 3.0, fps: str = "25", size: str = "320x240", audio: bool = True,
               extra: list[str] | None = None) -> Path:
    # A softened test pattern: real footage has smooth colour, testsrc2's raw edges don't.
    src = f"testsrc2=size={size}:rate={fps}:duration={seconds},gblur=sigma=1.5"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd += (extra or []) + [str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def small_video(tmp_path):
    return make_video(tmp_path / "in.mp4")


@pytest.fixture
def fake_engine():
    from tests.fakes import FakeEngine

    return FakeEngine()
