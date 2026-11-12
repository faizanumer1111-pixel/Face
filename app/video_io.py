"""Video probing, import, decoding and encoding with ffmpeg.

Frames move through ffmpeg pipes as raw BGR bytes, so phone rotation is applied by
ffmpeg and frame timing stays exact. Every imported video becomes a "working copy"
(work.mp4): H.264, constant frame rate, rotation applied, AAC audio. That keeps the
browser preview, frame numbers and renders in agreement.
"""

from __future__ import annotations

import functools
import json
import logging
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(50), Fraction(60000, 1001), Fraction(60)]
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
# Accurate YUV<->RGB conversion: frames that aren't swapped come back out of the
# encoder practically identical (about 60 dB PSNR before compression on real footage,
# vs about 41 dB with ffmpeg's defaults).
SWS_FLAGS = "accurate_rnd+full_chroma_int"


class VideoError(RuntimeError):
    """A video problem, worded for the person using the app."""


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise VideoError("ffmpeg isn't installed. Run scripts/setup.sh (it installs ffmpeg with Homebrew).")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise VideoError("ffprobe isn't installed. Run scripts/setup.sh (it installs ffmpeg with Homebrew).")
    return path


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> tuple[int, int]:
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-version"], capture_output=True, text=True).stdout
    try:
        ver = out.split()[2]
        parts = [int(p) for p in ver.replace("n", "").split("-")[0].split(".")[:2] if p.isdigit()]
        return (parts[0], parts[1] if len(parts) > 1 else 0)
    except (IndexError, ValueError):
        return (7, 0)


def fps_mode_args(mode: str) -> list[str]:
    return ["-fps_mode", mode] if ffmpeg_version() >= (5, 1) else ["-vsync", mode]


@functools.lru_cache(maxsize=1)
def _filters() -> str:
    return subprocess.run([ffmpeg_bin(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout


def has_filter(name: str) -> bool:
    return f" {name} " in _filters()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    total_frames: int
    duration: float
    has_audio: bool
    audio_codec: str | None = None
    codec: str = ""
    pix_fmt: str = ""
    matrix: str = "bt709"  # YCbCr matrix: "bt709" or "bt601"
    full_range: bool = False

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    def to_json(self) -> dict:
        d = asdict(self)
        d["fps"] = round(float(self.fps), 3)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "VideoInfo":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


def _ratio(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        r = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return r if r > 0 else None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    rot = (stream.get("tags") or {}).get("rotate")
    if rot:
        try:
            return int(rot) % 360
        except ValueError:
            pass
    return 0


def _matrix(stream: dict) -> str:
    return "bt601" if stream.get("color_space") in ("smpte170m", "bt470bg") else "bt709"


def _full_range(stream: dict) -> bool:
    return stream.get("pix_fmt", "").startswith("yuvj") or stream.get("color_range") == "pc"


def color_tags(matrix: str) -> list[str]:
    tag = "smpte170m" if matrix == "bt601" else "bt709"
    return ["-color_primaries", tag, "-color_trc", tag, "-colorspace", tag]


def probe_raw(path: Path) -> dict:
    proc = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise VideoError("Couldn't read this video file. Try exporting it as an MP4 or MOV.")
    data = json.loads(proc.stdout or "{}")
    if not any(s.get("codec_type") == "video" for s in data.get("streams", [])):
        raise VideoError("This file has no video in it.")
    return data


def _streams(data: dict) -> tuple[dict, dict | None]:
    vids = [s for s in data["streams"] if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    if not vids:
        raise VideoError("This file has no video in it.")
    auds = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    return vids[0], (auds[0] if auds else None)


def count_frames(path: Path) -> int:
    proc = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(proc.stdout.strip().split(",")[0])
    except ValueError:
        raise VideoError("Couldn't count the frames in this video.") from None


def probe(path: Path) -> VideoInfo:
    """Probe a working copy (CFR H.264), counting frames exactly."""
    data = probe_raw(path)
    v, a = _streams(data)
    r, avg = _ratio(v.get("r_frame_rate")), _ratio(v.get("avg_frame_rate"))
    if r and (not avg or abs(float(r) - float(avg)) / float(r) <= 0.01):
        fps = r
    else:
        fps = avg or r or Fraction(30)
    fps = fps.limit_denominator(1001)
    total = count_frames(path)
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or total / fps)
    return VideoInfo(
        width=int(v["width"]), height=int(v["height"]), fps_num=fps.numerator, fps_den=fps.denominator,
        total_frames=total, duration=duration, has_audio=a is not None,
        audio_codec=a.get("codec_name") if a else None, codec=v.get("codec_name", ""), pix_fmt=v.get("pix_fmt", ""),
        matrix=_matrix(v), full_range=_full_range(v),
    )


def frame_intervals(path: Path, seconds: int = 30) -> np.ndarray:
    """Gaps between video frame timestamps over the first `seconds`."""
    proc = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+{seconds}",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    pts = []
    for line in proc.stdout.splitlines():
        value = line.strip().strip(",")
        try:
            pts.append(float(value))
        except ValueError:
            continue
    return np.diff(np.sort(np.array(pts))) if len(pts) > 2 else np.array([])


def measured_vfr(intervals: np.ndarray) -> Fraction | None:
    """If the frame timing is irregular, the constant rate to convert to; else None."""
    gaps = intervals[intervals > 0]
    if gaps.size < 5:
        return None
    med = float(np.median(gaps))
    if float(np.percentile(gaps, 95) - np.percentile(gaps, 5)) <= 0.2 * med:
        return None
    return min(STANDARD_RATES, key=lambda s: abs(float(s) - 1.0 / med))


def choose_rate(r: Fraction | None, avg: Fraction | None) -> Fraction:
    """Pick a constant frame rate for a variable-frame-rate video."""
    target = avg or r or Fraction(30)
    if r and r <= 120 and abs(float(r) - float(target)) / float(target) < 0.02:
        target = r
    return min(STANDARD_RATES, key=lambda s: abs(float(s) - float(target)))


def prepare_source(src: Path, work: Path) -> tuple[VideoInfo, list[str]]:
    """Create the working copy of an uploaded video. Returns its info and notes for the UI."""
    data = probe_raw(src)
    v, a = _streams(data)
    r = _ratio(v.get("r_frame_rate"))
    avg = _ratio(v.get("avg_frame_rate"))
    codec = v.get("codec_name", "")
    pix_fmt = v.get("pix_fmt", "")
    rotation = _rotation(v)
    transfer = v.get("color_transfer", "")
    width, height = int(v.get("width", 0)), int(v.get("height", 0))
    vfr = bool(r and avg and abs(float(r) - float(avg)) / float(r) > 0.01)
    measured = measured_vfr(frame_intervals(src))
    vfr = vfr or measured is not None
    hdr = transfer in HDR_TRANSFERS
    odd = width % 2 == 1 or height % 2 == 1

    notes: list[str] = []
    reasons = []
    if codec != "h264":
        reasons.append("codec")
        notes.append(f"Converted the {codec.upper() or 'video'} video to H.264 so it plays in the browser.")
    if pix_fmt not in ("yuv420p", "yuvj420p") and not hdr:
        reasons.append("pix_fmt")
    if vfr:
        reasons.append("vfr")
    if rotation:
        reasons.append("rotation")
        notes.append("Applied the video's rotation so it's the right way up.")
    if hdr:
        reasons.append("hdr")
    if odd:
        reasons.append("odd")

    audio_args: list[str] = []
    if a is not None:
        audio_args = ["-c:a", "copy"] if a.get("codec_name") == "aac" else ["-c:a", "aac", "-b:a", "256k"]

    if not reasons:
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "copy", *audio_args, "-movflags", "+faststart", "-map_metadata", "-1", str(work)]
        _run(cmd, "Couldn't prepare this video")
    else:
        if vfr:
            rate = measured or choose_rate(r, avg)
        else:
            rate = (avg or r or Fraction(30)).limit_denominator(1001)
        filters = []
        if hdr:
            if has_filter("zscale") and has_filter("tonemap"):
                filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                               "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
                notes.append("Converted HDR footage to standard color.")
            else:
                notes.append("This is HDR footage; colors may look flatter. For best results export it as SDR.")
        if odd:
            filters.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
        filters.append("format=yuv420p")
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
               "-vf", ",".join(filters)]
        if vfr:
            cmd += ["-r", f"{rate.numerator}/{rate.denominator}", *fps_mode_args("cfr")]
            notes.append(f"Converted variable frame rate to a constant {float(rate):.3g} fps so the audio stays in sync.")
        gop = max(1, int(round(float(rate))))
        matrix = "bt709" if hdr else _matrix(v)
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-g", str(gop), *color_tags(matrix),
                *audio_args, "-movflags", "+faststart", "-map_metadata", "-1", str(work)]
        _run(cmd, "Couldn't convert this video")

    info = probe(work)
    if info.total_frames <= 0:
        raise VideoError("This video doesn't have any frames.")
    return info, notes


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise VideoError(f"{what}: {' '.join(tail) or 'ffmpeg failed'}")


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def seek_args(frame: int, fps: Fraction) -> list[str]:
    """Input seek that lands exactly on `frame` (ffmpeg drops frames before the time)."""
    if frame <= 0:
        return []
    t = (Fraction(frame) - Fraction(1, 2)) / fps
    return ["-ss", f"{float(t):.6f}"]


class FrameReader:
    """Decode `count` frames starting at `start` (all frames by default)."""

    def __init__(self, path: Path, info: VideoInfo, start: int = 0, count: int | None = None):
        self.info = info
        self.count = info.total_frames - start if count is None else count
        rng = "pc" if info.full_range else "tv"
        cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", *seek_args(start, info.fps), "-i", str(path),
               "-map", "0:v:0", "-frames:v", str(self.count),
               "-vf", f"scale=in_color_matrix={info.matrix}:in_range={rng}:flags={SWS_FLAGS},format=bgr24",
               *fps_mode_args("passthrough"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, bufsize=info.frame_bytes)

    def read(self) -> np.ndarray | None:
        n = self.info.frame_bytes
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                break
            got += k
        if got < n:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape((self.info.height, self.info.width, 3))

    def __iter__(self):
        for _ in range(self.count):
            frame = self.read()
            if frame is None:
                break
            yield frame

    def error_text(self) -> str:
        self._err.seek(0)
        return self._err.read().decode(errors="replace").strip()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if self.proc.stdout:
            self.proc.stdout.close()
        self._err.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_frame(path: Path, info: VideoInfo, index: int) -> np.ndarray:
    index = max(0, min(index, info.total_frames - 1))
    with FrameReader(path, info, start=index, count=1) as r:
        frame = r.read()
    if frame is None:
        raise VideoError(f"Couldn't read frame {index}.")
    return frame.copy()


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def videotoolbox_available() -> bool:
    if platform.system() != "Darwin":
        return False
    enc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" not in enc:
        return False
    test = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-frames:v", "10",
         "-c:v", "h264_videotoolbox", "-b:v", "2M", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return test.returncode == 0


def encoder_name(max_quality: bool) -> str:
    if not max_quality and videotoolbox_available():
        return "h264_videotoolbox"
    return "libx264"


def video_codec_args(info: VideoInfo, max_quality: bool) -> list[str]:
    if max_quality:
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "16"]
    if videotoolbox_available():
        bitrate = max(8_000_000, int(info.width * info.height * float(info.fps) * 0.2))
        return ["-c:v", "h264_videotoolbox", "-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.5)),
                "-bufsize", str(bitrate * 2), "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


class FrameWriter:
    """Encode raw frames from a pipe, adding audio from the source.

    `audio` is "full" (copy the source's audio), a list of (start, end) frame
    ranges to cut and join (clips mode), or None.
    """

    def __init__(self, out: Path, info: VideoInfo, source: Path, audio, max_quality: bool, log_path: Path):
        fps = info.fps
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
               "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        maps = ["-map", "0:v:0"]
        audio_codec: list[str] = []
        if info.has_audio and audio == "full":
            cmd += ["-i", str(source)]
            maps += ["-map", "1:a:0?"]
            audio_codec = ["-c:a", "copy"] if info.audio_codec == "aac" else ["-c:a", "aac", "-b:a", "256k"]
        elif info.has_audio and isinstance(audio, list) and audio:
            cmd += ["-i", str(source)]
            parts = []
            for k, (s, e) in enumerate(audio):
                ts, te = float(Fraction(s) / fps), float(Fraction(e) / fps)
                parts.append(f"[1:a:0]atrim=start={ts:.6f}:end={te:.6f},asetpts=PTS-STARTPTS[a{k}]")
            if len(audio) == 1:
                graph = parts[0].replace("[a0]", "[aout]")
            else:
                graph = ";".join(parts) + ";" + "".join(f"[a{k}]" for k in range(len(audio))) + \
                    f"concat=n={len(audio)}:v=0:a=1[aout]"
            cmd += ["-filter_complex", graph]
            maps += ["-map", "[aout]"]
            audio_codec = ["-c:a", "aac", "-b:a", "256k"]
        cmd += maps + ["-vf", f"scale=out_color_matrix={info.matrix}:out_range=tv:flags={SWS_FLAGS},format=yuv420p"]
        cmd += video_codec_args(info, max_quality) + [
            "-pix_fmt", "yuv420p", *color_tags(info.matrix), *audio_codec, "-movflags", "+faststart",
            "-f", "mp4", str(out)]
        self.log_path = log_path
        self._log = open(log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).data)
        except BrokenPipeError:
            raise VideoError(f"The video encoder stopped unexpectedly. Details: {self.log_path}") from None
        self.frames += 1

    def finish(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        self._log.close()
        if code != 0:
            raise VideoError(f"The video encoder failed. Details: {self.log_path}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        self._log.close()


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def load_image(path: Path, scratch: Path) -> np.ndarray:
    """Read a photo (JPG, PNG, WebP; HEIC via macOS `sips` or ffmpeg)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    converted = scratch / "converted.png"
    if platform.system() == "Darwin" and shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "png", str(path), "--out", str(converted)], capture_output=True)
    if not converted.exists():
        subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(path), "-frames:v", "1", str(converted)],
                       capture_output=True)
    img = cv2.imread(str(converted), cv2.IMREAD_COLOR) if converted.exists() else None
    if img is None:
        raise VideoError("Couldn't read this photo. Use a JPG or PNG.")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("Couldn't encode the image.")
    return buf.tobytes()


def face_crop(img: np.ndarray, bbox, pad: float = 0.35, size: int = 160) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + pad * 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[b:d, a:c]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
