"""Face detection, recognition and swapping on ONNX Runtime.

This re-implements the small part of InsightFace the app needs (SCRFD detection,
ArcFace embeddings, inswapper_128 swapping) so the app doesn't depend on the
insightface package, which has to be compiled from source on macOS. Outputs match
InsightFace 0.7.3; tests/test_engine_parity.py checks that when insightface is
installed.

The swap paste-back only touches the region around each face instead of the whole
frame, which keeps 1080p and 4K renders fast.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

DET_MODEL = "det_10g.onnx"
REC_MODEL = "w600k_r50.onnx"
SWAP_MODEL = "inswapper_128.onnx"
ENHANCE_MODEL = "gfpgan_1.4.onnx"  # sharper faces
OCCLUDE_MODEL = "xseg_1.onnx"  # hands / hair / objects in front of the face
MODEL_FILES = (DET_MODEL, REC_MODEL, SWAP_MODEL)
OPTIONAL_MODEL_FILES = (ENHANCE_MODEL, OCCLUDE_MODEL)
MASK_RES = 256  # blend masks are built at no more than this size, then scaled up
SHARPEN_MIN_FACE = 96  # faces shorter than this (pixels) are already as sharp as the swap gets

# Face template GFPGAN was trained on (FFHQ alignment), for a 512 px crop.
FFHQ_512 = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
) * 512


@dataclass
class SwapOptions:
    """How each face is swapped. Defaults are what looks most natural."""

    occlusion: bool = True  # keep hands, hair and objects in front of the face
    enhance: float = 0.6  # face sharpening strength, 0 = off
    color_match: bool = True  # match the new face to the scene's lighting

    @classmethod
    def from_json(cls, d: dict | None) -> "SwapOptions":
        d = d or {}
        return cls(occlusion=bool(d.get("occlusion", True)),
                   enhance=float(min(1.0, max(0.0, d.get("enhance", 0.6)))),
                   color_match=bool(d.get("color_match", True)))

# Five-point face template used by ArcFace and inswapper alignment.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Face:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in frame pixels
    kps: np.ndarray  # (5, 2) eyes, nose, mouth corners
    score: float
    embedding: np.ndarray | None = None  # L2-normalised (512,) when computed
    source: str | None = None  # id of the face to put on this one, set by the selectors

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def to_json(self) -> dict:
        return {"bbox": [round(float(v), 1) for v in self.bbox], "score": round(float(self.score), 3)}


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform src -> dst (same as skimage's SimilarityTransform)."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1
    t = np.eye(dim + 1, dtype=np.float64)
    u, s, v = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        return np.full((2, 3), np.nan)
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            t[:dim, :dim] = u @ v
        else:
            saved = d[dim - 1]
            d[dim - 1] = -1
            t[:dim, :dim] = u @ np.diag(d) @ v
            d[dim - 1] = saved
    else:
        t[:dim, :dim] = u @ np.diag(d) @ v
    scale = 1.0 / src_demean.var(axis=0).sum() * (s @ d)
    t[:dim, dim] = dst_mean - scale * (t[:dim, :dim] @ src_mean.T)
    t[:dim, :dim] *= scale
    return t[:2, :]


def estimate_norm(kps: np.ndarray, image_size: int) -> np.ndarray:
    if image_size % 112 == 0:
        ratio = image_size / 112.0
        diff_x = 0.0
    else:
        ratio = image_size / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    return umeyama(kps, dst)


def norm_crop(img: np.ndarray, kps: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    m = estimate_norm(kps, image_size)
    warped = cv2.warpAffine(img, m, (image_size, image_size), borderValue=0.0)
    return warped, m


# --------------------------------------------------------------------------- #
# ONNX Runtime sessions
# --------------------------------------------------------------------------- #

PROVIDER_CHOICES = ("coreml", "cuda", "cpu")


def available_provider_choices() -> list[str]:
    avail = set(ort.get_available_providers())
    out = []
    if "CoreMLExecutionProvider" in avail:
        out.append("coreml")
    if "CUDAExecutionProvider" in avail:
        out.append("cuda")
    out.append("cpu")
    return out


def _provider_attempts(choice: str, cache_dir: Path | None) -> list[list]:
    if choice == "coreml":
        opts = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            opts["ModelCacheDirectory"] = str(cache_dir)
        return [
            [("CoreMLExecutionProvider", opts), "CPUExecutionProvider"],
            [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}), "CPUExecutionProvider"],
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ]
    if choice == "cuda":
        return [["CUDAExecutionProvider", "CPUExecutionProvider"]]
    return [["CPUExecutionProvider"]]


def make_session(path: Path, choice: str = "cpu", cache_dir: Path | None = None) -> tuple[ort.InferenceSession, str]:
    """Create a session with the requested provider, falling back to CPU if it fails.

    Returns the session and the provider choice actually in use.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # Leave CPU cores free for video decoding/encoding, parallel swaps and the web page.
    so.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)
    if choice != "cpu":
        for providers in _provider_attempts(choice, cache_dir):
            try:
                sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
                active = sess.get_providers()
                if active and active[0] != "CPUExecutionProvider":
                    return sess, choice
            except Exception as exc:  # provider can't load this model
                log.warning("%s: %s provider failed (%s), trying the next option", path.name, choice, exc)
        log.warning("%s: %s not usable, using CPU", path.name, choice)
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess, "cpu"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class Detector:
    """SCRFD face detector (det_10g.onnx from InsightFace buffalo_l)."""

    def __init__(self, session: ort.InferenceSession, det_thresh: float = 0.5, nms_thresh: float = 0.4):
        self.session = session
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_name = session.get_inputs()[0].name
        outputs = session.get_outputs()
        self.output_names = [o.name for o in outputs]
        n = len(outputs)
        if n not in (6, 9, 10, 15):
            raise ValueError(f"Unexpected detector model with {n} outputs")
        self.fmc = 3 if n in (6, 9) else 5
        self.strides = [8, 16, 32] if self.fmc == 3 else [8, 16, 32, 64, 128]
        self.num_anchors = 2 if self.fmc == 3 else 1
        self.use_kps = n in (9, 15)
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def detect(self, img: np.ndarray, det_size: int = 640) -> list[Face]:
        h, w = img.shape[:2]
        im_ratio = h / w
        if im_ratio > 1.0:
            new_h = det_size
            new_w = int(new_h / im_ratio)
        else:
            new_w = det_size
            new_h = int(new_w * im_ratio)
        scale = new_h / h
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((det_size, det_size, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = resized
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, (det_size, det_size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]

        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx]
            bbox_preds = outs[idx + self.fmc] * stride
            fh, fw = det_size // stride, det_size // stride
            key = (fh, fw, stride)
            centers = self._centers.get(key)
            if centers is None:
                centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
                self._centers[key] = centers
            pos = np.where(scores >= self.det_thresh)[0]
            boxes = _distance2bbox(centers, bbox_preds)
            scores_l.append(scores[pos])
            boxes_l.append(boxes[pos])
            if self.use_kps:
                kps = _distance2kps(centers, outs[idx + self.fmc * 2] * stride)
                kps_l.append(kps.reshape((kps.shape[0], -1, 2))[pos])

        scores = np.vstack(scores_l).ravel()
        if scores.size == 0:
            return []
        order = scores.argsort()[::-1]
        boxes = np.vstack(boxes_l) / scale
        pre = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)[order]
        keep = _nms(pre, self.nms_thresh)
        dets = pre[keep]
        kpss = (np.vstack(kps_l) / scale)[order][keep] if self.use_kps else None
        faces = []
        for i in range(dets.shape[0]):
            kps = kpss[i] if kpss is not None else None
            faces.append(Face(bbox=dets[i, :4].copy(), kps=kps, score=float(dets[i, 4])))
        return faces


class Embedder:
    """ArcFace recognition model (w600k_r50.onnx from InsightFace buffalo_l)."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.output_names = [o.name for o in session.get_outputs()]

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        aimg, _ = norm_crop(img, face.kps, self.size)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 127.5, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(self.output_names, {self.input_name: blob})[0].ravel().astype(np.float32)
        emb /= max(float(np.linalg.norm(emb)), 1e-12)
        face.embedding = emb
        return emb


def load_emap(model_path: Path) -> np.ndarray:
    """The inswapper embedding map, cached next to the model after the first load."""
    cache = model_path.with_suffix(".emap.npy")
    if cache.exists():
        return np.load(cache)
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    inits = {init.name: init for init in model.graph.initializer}
    init = inits.get("buff2fs", model.graph.initializer[-1])
    emap = numpy_helper.to_array(init).astype(np.float32)
    if emap.shape != (512, 512):
        raise ValueError(f"Unexpected inswapper embedding map shape {emap.shape}")
    try:
        np.save(cache, emap)
    except OSError:
        pass
    return emap


class Swapper:
    """inswapper_128 face swapper with InsightFace's blending, limited to the face region."""

    def __init__(self, session: ort.InferenceSession, emap: np.ndarray):
        self.session = session
        self.emap = emap
        inputs = session.get_inputs()
        self.input_names = [i.name for i in inputs]
        self.output_names = [o.name for o in session.get_outputs()]
        self.size = int(inputs[0].shape[2])

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape((1, -1)).astype(np.float32) @ self.emap
        lat /= np.linalg.norm(lat)
        return lat.astype(np.float32)

    def swap(self, img: np.ndarray, face: Face, latent: np.ndarray, extra_mask: np.ndarray | None = None,
             color_match: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Swap `face` in `img` in place. Returns the swapped 128 px crop and its alignment."""
        size = self.size
        aimg, m = norm_crop(img, face.kps, size)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (size, size), (0.0, 0.0, 0.0), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        bgr_fake = np.clip(255 * pred.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[:, :, ::-1]
        if color_match:
            bgr_fake = match_lighting(bgr_fake, aimg)
        self.paste_back(img, bgr_fake, m, extra_mask)
        return bgr_fake, m

    @staticmethod
    def paste_back(img: np.ndarray, bgr_fake: np.ndarray, m: np.ndarray, extra_mask: np.ndarray | None = None) -> None:
        h, w = img.shape[:2]
        size = bgr_fake.shape[0]
        im = cv2.invertAffineTransform(m)
        corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], dtype=np.float64)
        pts = corners @ im.T
        # Mask size, erosion and blur match InsightFace's full-frame version; the
        # margin keeps the blur from reaching the edge of the region we work in.
        side = float(np.hypot(*(pts[1] - pts[0])))
        margin = int(np.ceil(side * 1.42 / 20)) + 6
        x0 = max(int(np.floor(pts[:, 0].min())) - margin, 0)
        y0 = max(int(np.floor(pts[:, 1].min())) - margin, 0)
        x1 = min(int(np.ceil(pts[:, 0].max())) + margin, w)
        y1 = min(int(np.ceil(pts[:, 1].max())) + margin, h)
        if x1 <= x0 or y1 <= y0:
            return
        rw, rh = x1 - x0, y1 - y0
        im_roi = im.copy()
        im_roi[0, 2] -= x0
        im_roi[1, 2] -= y0

        fake = cv2.warpAffine(bgr_fake, im_roi, (rw, rh), borderValue=0.0)
        # The blend mask is smooth, so it's built at no more than MASK_RES pixels and
        # scaled up. On a big 4K face that's ~20x less work for the same mask.
        sc = min(1.0, MASK_RES / max(rw, rh))
        sw, sh = max(1, round(rw * sc)), max(1, round(rh * sc))
        im_small = im_roi * sc
        white = np.full((size, size), 255, dtype=np.float32)
        mask = cv2.warpAffine(white, im_small, (sw, sh), borderValue=0.0)
        mask[mask > 20] = 255
        ys, xs = np.where(mask == 255)
        if ys.size == 0:
            return
        mask_size = np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())) / sc  # in full-size pixels
        k = max(1, round(max(int(mask_size) // 10, 10) * sc))
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        k = max(1, round(max(int(mask_size) // 20, 5) * sc))
        mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0) / 255.0
        if extra_mask is not None:
            mask *= cv2.warpAffine(extra_mask.astype(np.float32), im_small, (sw, sh), borderValue=0.0)
        if sc < 1.0:
            mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_LINEAR)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        roi = np.ascontiguousarray(img[y0:y1, x0:x1])
        img[y0:y1, x0:x1] = cv2.blendLinear(fake, roi, mask, 1.0 - mask)


_INNER = None


def _inner_face_mask(size: int) -> np.ndarray:
    """Ellipse over the cheeks, eyes and mouth of an aligned crop (no hair, no background)."""
    global _INNER
    if _INNER is None or _INNER.shape[0] != size:
        m = np.zeros((size, size), np.uint8)
        cv2.ellipse(m, (size // 2, int(size * 0.56)), (int(size * 0.28), int(size * 0.34)), 0, 0, 360, 1, -1)
        _INNER = m.astype(bool)
    return _INNER


def match_lighting(fake: np.ndarray, original: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """Shift the new face's brightness and colour cast towards the original face's.

    Works in LAB on the inner face only. Brightness follows the scene closely; colour
    moves only partway so the new person's skin tone is kept.
    """
    inner = _inner_face_mask(fake.shape[0])
    f = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    o = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = (o[inner].mean(axis=0) - f[inner].mean(axis=0)) * np.array([1.0, 0.35, 0.35], np.float32) * strength
    f += shift
    return cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def paste_crop(img: np.ndarray, crop: np.ndarray, m: np.ndarray, mask: np.ndarray) -> None:
    """Blend an aligned crop back into `img` (in place) through `mask` (crop coordinates, 0..1)."""
    h, w = img.shape[:2]
    size = crop.shape[0]
    im = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], dtype=np.float64) @ im.T
    x0, y0 = max(int(np.floor(corners[:, 0].min())) - 2, 0), max(int(np.floor(corners[:, 1].min())) - 2, 0)
    x1, y1 = min(int(np.ceil(corners[:, 0].max())) + 2, w), min(int(np.ceil(corners[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return
    im_roi = im.copy()
    im_roi[0, 2] -= x0
    im_roi[1, 2] -= y0
    rw, rh = x1 - x0, y1 - y0
    warped = cv2.warpAffine(crop, im_roi, (rw, rh), borderValue=0.0)
    sc = min(1.0, MASK_RES / max(rw, rh))  # smooth mask: build small, scale up
    a = cv2.warpAffine(mask.astype(np.float32), im_roi * sc, (max(1, round(rw * sc)), max(1, round(rh * sc))),
                       borderValue=0.0)
    if sc < 1.0:
        a = cv2.resize(a, (rw, rh), interpolation=cv2.INTER_LINEAR)
    a = np.ascontiguousarray(a, dtype=np.float32)
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    img[y0:y1, x0:x1] = cv2.blendLinear(warped, roi, a, 1.0 - a)


class Occluder:
    """XSeg occlusion model: which parts of an aligned face are actually visible."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.size = 256

    def mask(self, img: np.ndarray, kps: np.ndarray, size: int = 128) -> np.ndarray:
        """Visibility mask (1 = face, 0 = something in front) aligned like the `size` px swap crop."""
        crop, _ = norm_crop(img, kps, self.size)
        blob = (crop.astype(np.float32) / 255.0)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0, :, :, 0]
        out = cv2.GaussianBlur(out.clip(0, 1), (0, 0), 5).clip(0.5, 1)
        out = (out - 0.5) * 2
        return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


_BOX = None


def _box_mask(size: int = 512) -> np.ndarray:
    global _BOX
    if _BOX is None:
        blur = int(size * 0.5 * 0.3)
        edge = max(blur // 2, 1)
        m = np.ones((size, size), np.float32)
        m[:edge, :] = m[-edge:, :] = 0
        m[:, :edge] = m[:, -edge:] = 0
        _BOX = cv2.GaussianBlur(m, (0, 0), blur * 0.25)
    return _BOX


class Enhancer:
    """GFPGAN 1.4: restores detail in the 128 px swapped face."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def enhance(self, img: np.ndarray, kps: np.ndarray, strength: float,
                visible_128: np.ndarray | None = None, m128: np.ndarray | None = None) -> None:
        """Sharpen the face at `kps` in place. `visible_128` (with its alignment `m128`) keeps
        whatever is in front of the face untouched."""
        m = umeyama(kps, FFHQ_512)
        crop = cv2.warpAffine(img, m, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        blob = ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        out = ((np.clip(out, -1, 1) + 1) / 2).transpose(1, 2, 0)
        restored = (out * 255.0).round().astype(np.uint8)[:, :, ::-1]
        mask = _box_mask(512) * strength
        if visible_128 is not None and m128 is not None:
            # 128-crop coordinates -> 512-crop coordinates
            a = np.vstack([m128, [0, 0, 1]])
            b = np.vstack([m, [0, 0, 1]])
            to512 = (b @ np.linalg.inv(a))[:2]
            mask = mask * cv2.warpAffine(visible_128.astype(np.float32), to512, (512, 512), borderValue=0.0)
        paste_crop(img, restored, m, mask)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def settings_path(data_dir: Path) -> Path:
    return data_dir / "settings.json"


def load_provider_settings(data_dir: Path) -> dict[str, str]:
    p = settings_path(data_dir)
    if p.exists():
        try:
            return json.loads(p.read_text()).get("providers", {})
        except (OSError, ValueError):
            return {}
    return {}


def missing_models(models_dir: Path) -> list[str]:
    return [m for m in MODEL_FILES if not (models_dir / m).exists()]


class FaceEngine:
    """Loads the three models once and exposes detect / embed / swap."""

    def __init__(self, models_dir: Path, providers: dict[str, str] | None = None):
        missing = missing_models(models_dir)
        if missing:
            raise FileNotFoundError(
                "Missing model files: " + ", ".join(missing) + ". Run scripts/setup.sh to download them."
            )
        providers = providers or {}
        self.timings: dict[str, float] = {}
        self._timing_lock = threading.Lock()
        cache = models_dir / "coreml-cache"
        t0 = time.time()
        det_sess, self.det_provider = make_session(models_dir / DET_MODEL, providers.get("det", "cpu"), cache)
        rec_sess, self.rec_provider = make_session(models_dir / REC_MODEL, providers.get("rec", "cpu"), cache)
        swap_sess, self.swap_provider = make_session(models_dir / SWAP_MODEL, providers.get("swap", "cpu"), cache)
        self.detector = Detector(det_sess)
        self.embedder = Embedder(rec_sess)
        self.swapper = Swapper(swap_sess, load_emap(models_dir / SWAP_MODEL))
        self.enhancer = self.occluder = None
        self.enhance_provider = self.occlude_provider = None
        if (models_dir / ENHANCE_MODEL).exists():
            sess, self.enhance_provider = make_session(models_dir / ENHANCE_MODEL, providers.get("enhance", "cpu"), cache)
            self.enhancer = Enhancer(sess)
        if (models_dir / OCCLUDE_MODEL).exists():
            sess, self.occlude_provider = make_session(models_dir / OCCLUDE_MODEL, providers.get("occlude", "cpu"), cache)
            self.occluder = Occluder(sess)
        log.info("Models loaded in %.1fs (%s)", time.time() - t0, self.providers)

    def warm_up(self) -> None:
        """Run every model once so the first preview or render frame isn't the slow one.

        The Mac's GPU prepares each model on its first use, which can take several seconds.
        """
        img = np.full((360, 640, 3), 128, np.uint8)
        for size in (320, 640):
            self.detector.detect(img, size)
        kps = np.array([[240, 150], [290, 150], [265, 178], [245, 205], [285, 205]], np.float32)
        face = Face(bbox=np.array([200, 100, 330, 240], np.float32), kps=kps, score=1.0)
        self.embedder.embed(img, face)
        lat = self.latent(np.full(512, 1 / np.sqrt(512), np.float32))
        self.swap(img.copy(), face, lat, SwapOptions(occlusion=True, enhance=1.0, color_match=True))
        with self._timing_lock:
            self.timings.clear()

    @property
    def providers(self) -> dict[str, str]:
        out = {"det": self.det_provider, "rec": self.rec_provider, "swap": self.swap_provider}
        if self.enhancer:
            out["enhance"] = self.enhance_provider
        if self.occluder:
            out["occlude"] = self.occlude_provider
        return out

    @property
    def features(self) -> dict[str, bool]:
        return {"enhance": self.enhancer is not None, "occlusion": self.occluder is not None}

    # Time spent in each step, so a render can report what took longest.
    def _tick(self, step: str, t0: float) -> None:
        with self._timing_lock:
            self.timings[step] = self.timings.get(step, 0.0) + (time.perf_counter() - t0)

    def detect(self, img: np.ndarray, det_size: int = 640) -> list[Face]:
        t0 = time.perf_counter()
        try:
            return self.detector.detect(img, det_size)
        finally:
            self._tick("detect", t0)

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        t0 = time.perf_counter()
        try:
            return self.embedder.embed(img, face)
        finally:
            self._tick("match", t0)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        return self.swapper.latent(embedding)

    def swap(self, img: np.ndarray, face: Face, latent: np.ndarray, opts: SwapOptions | None = None) -> None:
        """Swap one face in place: occlusion mask, swap, lighting match, sharpening."""
        opts = opts or SwapOptions(occlusion=False, enhance=0.0, color_match=False)
        visible = None
        t0 = time.perf_counter()
        if opts.occlusion and self.occluder is not None:
            visible = self.occluder.mask(img, face.kps, self.swapper.size)
            self._tick("hands", t0)
            t0 = time.perf_counter()
        _, m128 = self.swapper.swap(img, face, latent, visible, opts.color_match)
        self._tick("swap", t0)
        # Small faces are already as sharp as the swap can make them; sharpening them only costs time.
        if opts.enhance > 0 and self.enhancer is not None and face.bbox[3] - face.bbox[1] >= SHARPEN_MIN_FACE:
            t0 = time.perf_counter()
            self.enhancer.enhance(img, face.kps, opts.enhance, visible, m128)
            self._tick("sharpen", t0)


def is_mac() -> bool:
    return platform.system() == "Darwin"
