"""VR and 3D video: formats, eyes, the stretch fix and the VR tags.

VR video stores each eye as an equirectangular image: a sphere unrolled flat, so
anything away from the centre is stretched. Swapping a face directly on that image
looks wrong, so each face is first rendered into a flat "virtual camera" view,
swapped there, and warped back. The two eyes are treated as one: the faces chosen in
the left eye decide what happens in the right eye.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FORMATS = {
    "flat": ("flat", "mono"),
    "flat_sbs": ("flat", "sbs"),
    "flat_tb": ("flat", "tb"),
    "vr180_sbs": ("equirect180", "sbs"),
    "vr180_tb": ("equirect180", "tb"),
    "vr180_mono": ("equirect180", "mono"),
    "vr360_mono": ("equirect360", "mono"),
    "vr360_tb": ("equirect360", "tb"),
    "vr360_sbs": ("equirect360", "sbs"),
}
FORMAT_LABELS = {
    "flat": "Normal video",
    "flat_sbs": "3D side-by-side",
    "flat_tb": "3D top-bottom",
    "vr180_sbs": "VR 180° side-by-side",
    "vr180_tb": "VR 180° top-bottom",
    "vr180_mono": "VR 180° (one eye)",
    "vr360_mono": "VR 360° (one eye)",
    "vr360_tb": "VR 360° top-bottom",
    "vr360_sbs": "VR 360° side-by-side",
}
FILENAME_TAGS = {
    "flat_sbs": "_3D_LR", "flat_tb": "_3D_TB", "vr180_sbs": "_180_LR", "vr180_tb": "_180_TB",
    "vr180_mono": "_180", "vr360_mono": "_360", "vr360_tb": "_360_TB", "vr360_sbs": "_360_LR",
}


@dataclass(frozen=True)
class Layout:
    projection: str = "flat"  # flat, equirect180, equirect360
    stereo: str = "mono"  # mono, sbs, tb

    @classmethod
    def of(cls, fmt: str) -> "Layout":
        proj, stereo = FORMATS.get(fmt, FORMATS["flat"])
        return cls(proj, stereo)

    @property
    def is_vr(self) -> bool:
        return self.projection != "flat"

    @property
    def is_flat_mono(self) -> bool:
        return self.projection == "flat" and self.stereo == "mono"


# --------------------------------------------------------------------------- #
# Detecting the format
# --------------------------------------------------------------------------- #


def detect_format(stream: dict, filename: str, width: int, height: int) -> tuple[str, str | None]:
    """Returns (format, reason). Format is "flat" unless metadata or the file name says otherwise."""
    projection = stereo = None
    for sd in stream.get("side_data_list", []) or []:
        kind = (sd.get("side_data_type") or "").lower()
        if "spherical" in kind:
            proj = (sd.get("projection") or "").lower()
            if "equirect" in proj:
                left = int(sd.get("bound_left", 0) or 0)
                right = int(sd.get("bound_right", 0) or 0)
                projection = "180" if left + right > 0 else "360"
        if "stereo" in kind:
            t = (sd.get("type") or "").lower()
            stereo = "sbs" if "side" in t else "tb" if ("top" in t or "bottom" in t) else None
    reason = "VR tags in the file" if projection or stereo else None

    name = Path(filename).stem.lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    if projection is None:
        if tokens & {"180", "vr180", "180x180", "180sbs", "180lr", "mkx200", "mkx220", "fisheye190"}:
            projection = "180"
        elif tokens & {"360", "vr360", "360x180"}:
            projection = "360"
        if projection:
            reason = "the file name"
    if stereo is None:
        if tokens & {"lr", "sbs", "3dh", "180lr", "180sbs"}:
            stereo = "sbs"
        elif tokens & {"tb", "ou", "3dv", "overunder"}:
            stereo = "tb"
        if stereo and not reason:
            reason = "the file name"

    if projection == "180":
        if stereo is None:
            stereo = "sbs" if width >= 2 * height * 0.95 else "mono"
        return f"vr180_{stereo}", reason
    if projection == "360":
        if stereo is None:
            stereo = "tb" if abs(width - height) < 0.05 * width else "mono"
        return f"vr360_{stereo}", reason
    if stereo in ("sbs", "tb"):
        return f"flat_{stereo}", reason
    return "flat", None


def looks_like_vr(width: int, height: int) -> bool:
    """A hint for the UI: very wide 2:1 or square high-res frames are often VR."""
    return width >= 3840 and (abs(width - 2 * height) < 0.02 * width or abs(width - height) < 0.02 * width)


# --------------------------------------------------------------------------- #
# Eyes
# --------------------------------------------------------------------------- #


def eyes(frame: np.ndarray, layout: Layout) -> list[np.ndarray]:
    """The eye images of a frame, as views into it (writing to them writes the frame)."""
    h, w = frame.shape[:2]
    if layout.stereo == "sbs":
        return [frame[:, : w // 2], frame[:, w // 2 : (w // 2) * 2]]
    if layout.stereo == "tb":
        return [frame[: h // 2], frame[h // 2 : (h // 2) * 2]]
    return [frame]


def eye_offset(layout: Layout, frame_shape: tuple, eye: int) -> tuple[int, int]:
    h, w = frame_shape[:2]
    if eye == 0:
        return 0, 0
    return (w // 2, 0) if layout.stereo == "sbs" else (0, h // 2)


def match_between_eyes(left: list, right: list) -> dict[int, object]:
    """Pair right-eye faces with left-eye faces (index in `left` -> right face).

    The same person sits at almost the same place in both eyes, shifted sideways by
    the stereo offset, so the nearest face of similar size is the match.
    """
    pairs: dict[int, object] = {}
    used: set[int] = set()
    order = sorted(range(len(left)), key=lambda i: -left[i].area)
    for i in order:
        lf = left[i]
        lw = max(1.0, float(lf.bbox[2] - lf.bbox[0]))
        lh = max(1.0, float(lf.bbox[3] - lf.bbox[1]))
        best, best_d = None, None
        for j, rf in enumerate(right):
            if j in used:
                continue
            dx = abs(rf.center[0] - lf.center[0]) / lw
            dy = abs(rf.center[1] - lf.center[1]) / lh
            size = rf.area / max(lf.area, 1.0)
            if dx < 1.2 and dy < 0.5 and 0.5 < size < 2.0:
                d = dx + 2 * dy
                if best_d is None or d < best_d:
                    best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs[i] = right[best]
    return pairs


# --------------------------------------------------------------------------- #
# The stretch fix: flat views of an equirectangular eye
# --------------------------------------------------------------------------- #


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


class EquirectView:
    """A flat virtual-camera view looking at (yaw, pitch) inside an equirectangular eye."""

    def __init__(self, eye_shape: tuple, projection: str, yaw: float, pitch: float, fov: float, size: int):
        self.h, self.w = eye_shape[:2]
        self.span = math.pi if projection == "equirect180" else 2 * math.pi
        self.yaw, self.pitch, self.fov, self.size = yaw, pitch, fov, size
        self.r = _rotation(yaw, pitch)
        self.f = (size / 2) / math.tan(fov / 2)

    # equirect pixel <-> direction
    def _to_eq(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = np.arctan2(d[..., 0], d[..., 2])
        lat = np.arcsin(np.clip(d[..., 1], -1, 1))
        x = (lon + self.span / 2) / self.span * self.w - 0.5
        y = (math.pi / 2 - lat) / math.pi * self.h - 0.5
        return x.astype(np.float32), y.astype(np.float32)

    def _dirs(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lon = (x + 0.5) / self.w * self.span - self.span / 2
        lat = math.pi / 2 - (y + 0.5) / self.h * math.pi
        return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)

    def render(self, eye: np.ndarray) -> np.ndarray:
        s = self.size
        u, v = np.meshgrid(np.arange(s, dtype=np.float64) + 0.5 - s / 2, np.arange(s, dtype=np.float64) + 0.5 - s / 2)
        cam = np.stack([u, -v, np.full_like(u, self.f)], axis=-1)
        cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
        world = cam @ self.r.T
        mx, my = self._to_eq(world)
        border = cv2.BORDER_WRAP if self.span > 4 else cv2.BORDER_REPLICATE
        return cv2.remap(eye, mx, my, cv2.INTER_LINEAR, borderMode=border)

    def footprint(self, region: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
        """Bounding box in the eye of a region of the view (default: the whole view).

        Faces that straddle the 360° seam or a pole get the full width of their rows.
        """
        u0, v0, u1, v1 = region or (0, 0, self.size, self.size)
        t = np.linspace(0.0, 1.0, 24)
        us = np.concatenate([u0 + t * (u1 - u0), u0 + t * (u1 - u0), np.full_like(t, u0), np.full_like(t, u1)])
        vs = np.concatenate([np.full_like(t, v0), np.full_like(t, v1), v0 + t * (v1 - v0), v0 + t * (v1 - v0)])
        s = self.size
        cam = np.stack([us - s / 2, -(vs - s / 2), np.full_like(us, self.f)], axis=-1)
        cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
        x, y = self._to_eq(cam @ self.r.T)
        y0, y1 = max(int(y.min()) - 2, 0), min(int(y.max()) + 3, self.h)
        if x.max() - x.min() > self.w * 0.5:  # wraps the seam or goes over a pole
            return 0, y0, self.w, y1
        return max(int(x.min()) - 2, 0), y0, min(int(x.max()) + 3, self.w), y1

    def paste(self, eye: np.ndarray, original_view: np.ndarray, edited_view: np.ndarray) -> None:
        """Warp the changed pixels of `edited_view` back into the eye (in place)."""
        diff = np.abs(edited_view.astype(np.int16) - original_view.astype(np.int16)).max(axis=2)
        changed = (diff > 1).astype(np.uint8)
        if not changed.any():
            return
        changed = cv2.dilate(changed, np.ones((5, 5), np.uint8))
        alpha = cv2.GaussianBlur(changed.astype(np.float32), (7, 7), 0)
        vs, us = np.nonzero(changed)
        region = (max(int(us.min()) - 6, 0), max(int(vs.min()) - 6, 0),
                  min(int(us.max()) + 7, self.size), min(int(vs.max()) + 7, self.size))
        x0, y0, x1, y1 = self.footprint(region)
        if x1 <= x0 or y1 <= y0:
            return
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
        cam = self._dirs(xs, ys) @ self.r  # world -> camera (inverse rotation)
        z = cam[..., 2]
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        u = (self.f * cam[..., 0] / zs + self.size / 2 - 0.5).astype(np.float32)
        v = (-self.f * cam[..., 1] / zs + self.size / 2 - 0.5).astype(np.float32)
        u[~ok] = -10
        back = cv2.remap(edited_view, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.remap(alpha, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[:, :, None]
        roi = eye[y0:y1, x0:x1].astype(np.float32)
        eye[y0:y1, x0:x1] = (a * back.astype(np.float32) + (1 - a) * roi).astype(np.uint8)


def view_for_face(eye_shape: tuple, projection: str, bbox: np.ndarray, size_hint: int | None = None) -> EquirectView:
    """A view centred on a face found in the equirectangular eye, wide enough to hold its head."""
    h, w = eye_shape[:2]
    span = math.pi if projection == "equirect180" else 2 * math.pi
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    yaw = (cx + 0.5) / w * span - span / 2
    pitch = math.pi / 2 - (cy + 0.5) / h * math.pi
    ang = max(1e-3, (bbox[3] - bbox[1]) / h * math.pi)  # vertical size isn't stretched
    fov = min(math.radians(110), max(math.radians(15), ang * 3.0))
    if size_hint is None:
        size_hint = int(np.clip((bbox[3] - bbox[1]) * 3.0, 256, 1024))
    size = int(math.ceil(size_hint / 32) * 32)
    return EquirectView(eye_shape, projection, yaw, pitch, fov, size)


# --------------------------------------------------------------------------- #
# VR tags in MP4 files (Spherical Video V2: st3d + sv3d)
# --------------------------------------------------------------------------- #


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full_box(kind: bytes, payload: bytes) -> bytes:
    return _box(kind, b"\x00\x00\x00\x00" + payload)


def vr_boxes(fmt: str) -> bytes:
    layout = Layout.of(fmt)
    stereo_mode = {"mono": 0, "tb": 1, "sbs": 2}[layout.stereo]
    out = _full_box(b"st3d", bytes([stereo_mode]))
    if layout.is_vr:
        side = 0x40000000 if layout.projection == "equirect180" else 0
        svhd = _full_box(b"svhd", b"FaceSwap Studio\x00")
        prhd = _full_box(b"prhd", struct.pack(">iii", 0, 0, 0))
        equi = _full_box(b"equi", struct.pack(">IIII", 0, 0, side, side))
        out += _box(b"sv3d", svhd + _box(b"proj", prhd + equi))
    return out


def _children(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            header = 16
        elif size == 0:
            size = end - i
        yield kind, i, header, size
        i += size


def inject_vr_tags(path: Path, fmt: str) -> bool:
    """Add VR tags to an MP4 whose moov box is at the end (written without faststart).

    Inserts st3d/sv3d into the video sample entry and fixes the sizes of every box
    around it. Because moov comes after mdat, no media offsets move.
    """
    if fmt == "flat":
        return False
    data = bytearray(path.read_bytes())
    top = list(_children(data, 0, len(data)))
    moov = next((t for t in top if t[0] == b"moov"), None)
    if moov is None or moov[1] < next((t[1] for t in top if t[0] == b"mdat"), 0):
        return False
    _, m_off, m_hdr, m_size = moov
    for kind, t_off, t_hdr, t_size in _children(data, m_off + m_hdr, m_off + m_size):
        if kind != b"trak":
            continue
        chain = [(t_off, t_hdr)]
        found = _find_entry(data, t_off + t_hdr, t_off + t_size, chain)
        if found is None:
            continue
        e_off, e_hdr, e_size = found
        extra = vr_boxes(fmt)
        insert_at = e_off + e_size
        data[insert_at:insert_at] = extra
        for off, hdr in chain + [(m_off, m_hdr)]:
            if hdr == 8:
                size = struct.unpack(">I", data[off:off + 4])[0]
                data[off:off + 4] = struct.pack(">I", size + len(extra))
            else:
                size = struct.unpack(">Q", data[off + 8:off + 16])[0]
                data[off + 8:off + 16] = struct.pack(">Q", size + len(extra))
        path.write_bytes(bytes(data))
        return True
    return False


def _find_entry(data: bytearray, start: int, end: int, chain: list):
    """Walk trak > mdia > minf > stbl > stsd > (avc1|hvc1|hev1), recording ancestors."""
    path = [b"mdia", b"minf", b"stbl", b"stsd"]
    s, e = start, end
    for want in path:
        hit = next(((o, h, z) for k, o, h, z in _children(data, s, e) if k == want), None)
        if hit is None:
            return None
        o, h, z = hit
        chain.append((o, h))
        s, e = o + h, o + z
        if want == b"stsd":
            s += 8  # version/flags + entry count
    for k, o, h, z in _children(data, s, e):
        if k in (b"avc1", b"hvc1", b"hev1"):
            chain.append((o, h))
            # the new boxes go at the end of the sample entry, so the entry itself grows too
            return o, h, z
        return None
    return None
