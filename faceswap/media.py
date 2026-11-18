"""ffmpeg helpers: probing, frame-accurate raw frame streaming, encoding, concat + audio mux."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
from fractions import Fraction

import cv2
import numpy as np

import shutil as _shutil


def _find_ffmpeg() -> tuple[str, str]:
    """ffmpeg + ffprobe from PATH, or from the static-ffmpeg package (no Homebrew needed on a Mac)."""
    ff, fp = os.environ.get("FFMPEG"), os.environ.get("FFPROBE")
    if ff and fp:
        return ff, fp
    if _shutil.which("ffmpeg") and _shutil.which("ffprobe"):
        return "ffmpeg", "ffprobe"
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        if _shutil.which("ffmpeg") and _shutil.which("ffprobe"):
            return "ffmpeg", "ffprobe"
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError("ffmpeg was not found. Install it (brew install ffmpeg) or run: pip install static-ffmpeg")


FFMPEG, FFPROBE = _find_ffmpeg()
BASE = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, **kw)
    if p.returncode != 0:
        err = p.stderr.decode(errors="replace").strip().splitlines()
        raise MediaError(err[-1] if err else f"{cmd[0]} failed")
    return p


def _rate(s: str | None) -> Fraction | None:
    try:
        r = Fraction(s)
        return r if 0 < r <= 240 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe(path: str) -> dict:
    out = _run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MediaError("That file has no video track.")
    rate = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")) or Fraction(30)
    rate = rate.limit_denominator(1001)
    w, h = int(video["width"]), int(video["height"])
    rot = 0
    try:
        rot = int(float(video.get("tags", {}).get("rotate", 0)))
    except ValueError:
        pass
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    duration = float(video.get("duration") or info.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise MediaError("Could not read the video's length.")
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return {
        "width": w, "height": h, "fps": float(rate), "fps_str": f"{rate.numerator}/{rate.denominator}",
        "duration": duration, "frames": max(1, int(round(duration * float(rate)))),
        "has_audio": has_audio, "codec": video.get("codec_name"),
    }


def _seek_args(start_frame: int, fps: float) -> list[str]:
    # half a frame early so float rounding never skips the first wanted frame
    return ["-ss", f"{(start_frame - 0.5) / fps:.6f}"] if start_frame > 0 else []


class FrameReader:
    """Streams decoded BGR frames [start_frame, start_frame + count) through a pipe,
    prefetching on a background thread."""

    def __init__(self, path: str, meta: dict, start_frame: int, count: int, prefetch: int = 12):
        self.w, self.h = meta["width"], meta["height"]
        self.count = count
        cmd = BASE + _seek_args(start_frame, meta["fps"]) + [
            # setpts puts the first wanted frame at t=0 so fps= never duplicates it
            "-i", path, "-map", "0:v:0", "-vf", f"setpts=PTS-STARTPTS,fps={meta['fps_str']}", "-frames:v", str(count),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, bufsize=0)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        size = self.w * self.h * 3
        try:
            for _ in range(self.count):
                buf = np.empty((self.h, self.w, 3), np.uint8)
                mv, got = memoryview(buf).cast("B"), 0
                while got < size:
                    n = self.proc.stdout.readinto(mv[got:])
                    if not n:
                        break
                    got += n
                if got < size:
                    break
                self.q.put(buf)
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            yield f
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FrameWriter:
    """Encodes BGR frames to H.264 on a background thread."""

    def __init__(self, path: str, meta: dict, crf: int, preset: str):
        self.path = path
        cmd = BASE + ["-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{meta['width']}x{meta['height']}",
                      "-r", meta["fps_str"], "-i", "-", "-an",
                      "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                      "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                      "-movflags", "+faststart", path]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err, bufsize=0)
        self.q: queue.Queue = queue.Queue(maxsize=12)
        self.error: Exception | None = None
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            if self.error:
                continue
            try:
                self.proc.stdin.write(memoryview(np.ascontiguousarray(f)).cast("B"))
            except (BrokenPipeError, OSError) as e:
                self.error = e

    def write(self, frame: np.ndarray):
        self.q.put(frame)

    def close(self):
        self.q.put(None)
        self.t.join()
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0 or self.error:
            self._err.seek(0)
            raise MediaError("Encoder failed: " + self._err.read().decode(errors="replace")[-400:])


def grab_frame(path: str, meta: dict, t: float) -> np.ndarray:
    """One decoded frame at time t (seconds)."""
    t = min(max(0.0, t), max(0.0, meta["duration"] - 1.0 / meta["fps"]))
    for tt in (t, max(0.0, t - 0.5), 0.0):
        p = subprocess.run(BASE + ["-ss", f"{tt:.3f}", "-i", path, "-map", "0:v:0", "-frames:v", "1",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        size = meta["width"] * meta["height"] * 3
        if len(p.stdout) >= size:
            return np.frombuffer(p.stdout[:size], np.uint8).reshape(meta["height"], meta["width"], 3).copy()
    raise MediaError(f"Could not read a frame at {t:.2f}s")


def frame_jpeg(path: str, t: float, width: int = 240) -> bytes:
    """Small JPEG thumbnail at time t straight from ffmpeg (used for the timeline filmstrip)."""
    for tt in (t, max(0.0, t - 0.5), 0.0):
        p = subprocess.run(BASE + ["-ss", f"{max(0.0, tt):.3f}", "-i", path, "-map", "0:v:0", "-frames:v", "1",
                                   "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                           capture_output=True)
        if p.stdout:
            return p.stdout
    raise MediaError(f"Could not read a frame at {t:.2f}s")


def jpeg(img: np.ndarray, max_w: int = 1280, quality: int = 88) -> bytes:
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise MediaError("JPEG encode failed")
    return buf.tobytes()


def concat_and_mux(chunks: list[str], source: str, meta: dict, start: float, duration: float, out: str) -> None:
    """Join rendered chunks (stream copy) and lay the original audio for the same range back on."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for c in chunks:
            lst.write(f"file '{os.path.abspath(c)}'\n")
        list_path = lst.name
    try:
        cmd = BASE + ["-y", "-f", "concat", "-safe", "0", "-i", list_path]
        if meta.get("has_audio"):
            cmd += ["-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", source,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += ["-c:v", "copy", "-t", f"{duration:.6f}", "-movflags", "+faststart", out]
        _run(cmd)
    finally:
        os.unlink(list_path)
