"""Video probing, import, decoding and encoding with ffmpeg.

Frames move through ffmpeg pipes as raw BGR bytes, so phone rotation is applied by
ffmpeg and frame timing stays exact. Every imported video becomes a "working copy"
(work.mp4): H.264, constant frame rate, rotation applied, AAC audio. That keeps the
browser preview, frame numbers and renders in agreement.
"""

from __future__ import annotations

import functools
import json
import os
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


@functools.lru_cache(maxsize=1)
def hw_decode_args() -> tuple[str, ...]:
    """["-hwaccel", "videotoolbox"] when hardware decoding is switched on, else nothing.

    Off unless FACESWAP_HWDECODE=1: software decoding is the setting proven on real Macs.
    """
    if platform.system() != "Darwin" or os.environ.get("FACESWAP_HWDECODE") != "1":
        return ()
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "t.mp4"
        made = subprocess.run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
                               "-frames:v", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                              capture_output=True)
        if made.returncode != 0:
            return ()
        out = subprocess.run([ffmpeg_bin(), "-v", "error", "-hwaccel", "videotoolbox", "-i", str(clip),
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        ok = out.returncode == 0 and len(out.stdout) == 5 * 320 * 240 * 3
    return ("-hwaccel", "videotoolbox") if ok else ()


@functools.lru_cache(maxsize=4)
def hw_encoder_works(name: str) -> bool:
    if platform.system() != "Darwin":
        return False
    test = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-frames:v", "10",
         "-c:v", name, "-b:v", "2M", "-f", "null", "-"], capture_output=True)
    return test.returncode == 0


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
    vr_format: str = "flat"  # detected layout, see vr.FORMATS
    vr_reason: str | None = None  # why we think it's VR/3D
    vr_hint: bool = False  # big 2:1 or square frame: might be VR
    has_proxy: bool = False  # a small browser copy exists (proxy.mp4)
    needs_proxy: bool = False  # big or not H.264: the browser should get a small copy

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
    if codec not in ("h264", "hevc"):
        reasons.append("codec")
        notes.append(f"Converted the {codec.upper() or 'video'} video to H.264.")
    if codec == "h264" and pix_fmt not in ("yuv420p", "yuvj420p") and not hdr:
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

    from .vr import detect_format, looks_like_vr

    vr_format, vr_reason = detect_format(v, src.name, width, height)
    big = width * height > 4096 * 2304

    if not reasons:
        tag = ["-tag:v", "hvc1"] if codec == "hevc" else []
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "copy", *tag, *audio_args, "-movflags", "+faststart", "-map_metadata", "-1", str(work)]
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
        if big:  # past H.264's comfortable size (8K VR): keep it HEVC
            enc = ["-c:v", "libx265", "-preset", "fast", "-crf", "14", "-tag:v", "hvc1",
                   "-x265-params", f"keyint={gop}:log-level=error"]
        else:
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-g", str(gop)]
        tail = [*color_tags(matrix), *audio_args, "-movflags", "+faststart", "-map_metadata", "-1", str(work)]
        hw_name = "hevc_videotoolbox" if big else "h264_videotoolbox"
        if hw_encoder_works(hw_name):
            # The Mac's media engine converts several times faster than the CPU. It's only an
            # in-between copy, so a generous bitrate keeps it visually lossless.
            bitrate = int(min(200e6, max(12e6, width * height * float(rate) * (0.25 if big else 0.35))))
            hw_cmd = [ffmpeg_bin(), "-y", "-v", "error", *hw_decode_args(), *cmd[4:],
                      "-c:v", hw_name, "-b:v", str(bitrate), "-g", str(gop),
                      *(["-tag:v", "hvc1"] if big else []), *tail]
            if subprocess.run(hw_cmd, capture_output=True).returncode == 0:
                cmd = None
        if cmd is not None:
            _run(cmd + enc + tail, "Couldn't convert this video")

    info = probe(work)
    if info.total_frames <= 0:
        raise VideoError("This video doesn't have any frames.")
    info.vr_format, info.vr_reason = vr_format, vr_reason
    info.vr_hint = vr_format == "flat" and looks_like_vr(width, height)
    if vr_format != "flat":
        from .vr import FORMAT_LABELS

        notes.append(f"Looks like {FORMAT_LABELS[vr_format]} (from {vr_reason}). You can change this under Video format.")
    elif info.vr_hint:
        notes.append("This might be VR footage. If it is, choose the right one under Video format.")
    if info.codec != "h264" or max(info.width, info.height) > 2560:
        info.needs_proxy = True  # made in the background by the app, so importing returns at once
    return info, notes


def make_proxy(work: Path, proxy: Path, info: VideoInfo) -> None:
    """A light H.264 copy for the browser: every frame kept (so frame numbers match), max 1920 px."""
    scale = "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))'"
    gop = max(1, int(round(float(info.fps))))
    head = ["-i", str(work), "-map", "0:v:0", "-map", "0:a:0?", "-vf", scale + ",format=yuv420p"]
    tail = ["-g", str(gop), *fps_mode_args("passthrough"), "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-f", "mp4", str(proxy) + ".part"]
    if hw_encoder_works("h264_videotoolbox"):
        # Mac media engine for both decoding and encoding: much faster for 4K and 8K VR.
        proc = subprocess.run([ffmpeg_bin(), "-y", "-v", "error", *hw_decode_args(), *head,
                               "-c:v", "h264_videotoolbox", "-b:v", "8M", *tail], capture_output=True)
        if proc.returncode == 0:
            Path(str(proxy) + ".part").replace(proxy)
            return
    _run([ffmpeg_bin(), "-y", "-v", "error", *head, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", *tail],
         "Couldn't make the preview copy")
    Path(str(proxy) + ".part").replace(proxy)


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
        self.path, self.start = path, start
        self.read_any = False
        self.hw = bool(hw_decode_args())
        self._start_ffmpeg()

    def _start_ffmpeg(self) -> None:
        info = self.info
        rng = "pc" if info.full_range else "tv"
        hw = list(hw_decode_args()) if self.hw else []
        cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", *hw, *seek_args(self.start, info.fps), "-i", str(self.path),
               "-map", "0:v:0", "-frames:v", str(self.count),
               "-vf", f"scale=in_color_matrix={info.matrix}:in_range={rng}:flags={SWS_FLAGS},format=bgr24",
               *fps_mode_args("passthrough"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, bufsize=info.frame_bytes)

    def read(self) -> np.ndarray | None:
        frame = self._read()
        if frame is None and not self.read_any and self.hw:
            # The hardware decoder couldn't handle this file: start again in software.
            log.warning("Hardware decoding failed (%s); using software decoding", self.error_text()[-200:])
            self.close()
            self.hw = False
            self._start_ffmpeg()
            frame = self._read()
        if frame is not None:
            self.read_any = True
        return frame

    def _read(self) -> np.ndarray | None:
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


@functools.lru_cache(maxsize=1)
def hevc_videotoolbox_available() -> bool:
    if platform.system() != "Darwin":
        return False
    test = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-frames:v", "10",
         "-c:v", "hevc_videotoolbox", "-b:v", "2M", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return test.returncode == 0


def use_hevc(info: VideoInfo, vr: bool) -> bool:
    return vr or info.width * info.height > 4096 * 2304


def hevc_codec_args(info: VideoInfo, max_quality: bool) -> list[str]:
    if not max_quality and hevc_videotoolbox_available():
        bitrate = int(min(150e6, max(10e6, info.width * info.height * float(info.fps) * 0.06)))
        return ["-c:v", "hevc_videotoolbox", "-b:v", str(bitrate), "-tag:v", "hvc1"]
    preset, crf = ("slow", "18") if max_quality else ("fast", "20")
    return ["-c:v", "libx265", "-preset", preset, "-crf", crf, "-tag:v", "hvc1", "-x265-params", "log-level=error"]


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

    def __init__(self, out: Path, info: VideoInfo, source: Path, audio, max_quality: bool, log_path: Path,
                 vr: bool = False, index_at_end: bool = False):
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
        codec = hevc_codec_args(info, max_quality) if use_hevc(info, vr) else video_codec_args(info, max_quality)
        # VR files keep the index at the end for now so the VR tags can be added afterwards.
        flags = [] if (vr or index_at_end) else ["-movflags", "+faststart"]
        cmd += codec + ["-pix_fmt", "yuv420p", *color_tags(info.matrix), *audio_codec, *flags, "-f", "mp4", str(out)]
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


def finalize_vr(tmp: Path, out: Path, vr_format: str) -> None:
    """Add the VR/3D tags so headsets and YouTube recognise the file, then move the index to the front."""
    from .vr import inject_vr_tags

    inject_vr_tags(tmp, vr_format)
    proc = subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(tmp), "-map", "0", "-c", "copy",
                           "-strict", "unofficial", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.replace(out)  # still a valid video, just without the index moved
    else:
        tmp.unlink(missing_ok=True)


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
