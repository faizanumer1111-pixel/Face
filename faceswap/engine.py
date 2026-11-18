"""Face engine: detection, identity matching, swapping, restoration and blending.

Speed notes
-----------
* All blending happens on a small region of interest around the face instead of
  warping full-resolution masks (the stock insightface paste-back does that and
  it dominates runtime on 1080p/4K footage).
* Detection + recognition are stateless and thread-safe, so the pipeline runs
  them ahead of time in a thread pool while the swap stage works on the previous
  frame. ONNX Runtime releases the GIL, so the stages genuinely overlap.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .config import MATCH_THRESHOLD, MODEL_DIR, TRACK_THRESHOLD, Tier

# GFPGAN / FFHQ alignment template for a 512px crop (facexlib face_template)
FFHQ_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)

# BiSeNet (CelebAMask-HQ) classes that belong to the face itself.
# Everything else - hair, hands, microphones, background - stays in front.
FACE_REGIONS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Face:
    bbox: np.ndarray               # x1, y1, x2, y2
    kps: np.ndarray                # 5 x 2 landmarks
    score: float
    emb: np.ndarray | None = None  # L2-normalised ArcFace embedding

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(x2 - x1, y2 - y1))

    def offset(self, dx: float) -> "Face":
        return Face(self.bbox + np.array([dx, 0, dx, 0], np.float32), self.kps + np.array([dx, 0], np.float32),
                    self.score, self.emb)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def box_mask(size: int, blur: float = 0.3) -> np.ndarray:
    """Feathered rectangle mask (float32 0..1) for blending a square crop."""
    blur_amount = int(size * 0.5 * blur)
    edge = max(blur_amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:edge, :] = 0
    m[-edge:, :] = 0
    m[:, :edge] = 0
    m[:, -edge:] = 0
    if blur_amount > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
    return m


def paste_roi(frame: np.ndarray, crop: np.ndarray, M: np.ndarray, mask: np.ndarray) -> None:
    """Blend `crop` (aligned with affine M: frame -> crop) back into `frame` in place,
    touching only the bounding box of the warped crop."""
    H, W = frame.shape[:2]
    ch, cw = crop.shape[:2]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [cw, 0, 1], [0, ch, 1], [cw, ch, 1]], np.float32) @ IM.T
    x0 = max(int(math.floor(corners[:, 0].min())) - 1, 0)
    y0 = max(int(math.floor(corners[:, 1].min())) - 1, 0)
    x1 = min(int(math.ceil(corners[:, 0].max())) + 1, W)
    y1 = min(int(math.ceil(corners[:, 1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    IMr = IM.copy()
    IMr[0, 2] -= x0
    IMr[1, 2] -= y0
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(crop, IMr, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, IMr, size, flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(m * warped.astype(np.float32) + (1.0 - m) * roi, 0, 255).astype(np.uint8)


def match_low_frequencies(img: np.ndarray, ref: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Keep the fine detail of `img` but take overall colour/lighting from `ref`.
    Stops face restoration from shifting skin tone away from the swap."""
    small = (32, 32)
    d = cv2.resize(ref, small, interpolation=cv2.INTER_AREA).astype(np.float32) - \
        cv2.resize(img, small, interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(img.astype(np.float32) + d * strength, 0, 255).astype(np.uint8)


class OneEuro:
    """One-Euro filter for landmark jitter: smooth when still, responsive when moving."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        tau = 1.0 / (2 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


@dataclass
class TargetTrack:
    """Follows one chosen person through the shot."""
    target: np.ndarray                 # ArcFace centroid of the person in the video
    smoothing: float
    fps: float
    last_bbox: np.ndarray | None = None
    last_idx: int = -10_000
    filt: OneEuro | None = None

    verified_idx: int = -10_000

    def _recent(self, idx: int) -> bool:
        return idx - self.last_idx <= max(3, int(self.fps * 0.25))

    def locked(self, faces: list[Face], idx: int) -> int | None:
        """Index of the one face that clearly continues this track (by overlap alone), or None.
        Lets us skip identity checks on most frames: they only run when the lock is lost,
        when another face comes close, or periodically to re-verify."""
        if not self._recent(idx) or self.last_bbox is None:
            return None
        ovs = [iou(f.bbox, self.last_bbox) for f in faces]
        close = [i for i, o in enumerate(ovs) if o >= 0.2]
        if len(close) == 1 and ovs[close[0]] >= 0.4:
            return close[0]
        return None

    def score(self, f: Face, idx: int) -> float | None:
        """How well `f` matches this person (None = not them). Overlap with where the
        person just was keeps the lock through profile turns, blur and partial occlusion."""
        sim = float(f.emb @ self.target) if f.emb is not None else 0.0
        ov = iou(f.bbox, self.last_bbox) if self._recent(idx) else 0.0
        if sim >= MATCH_THRESHOLD or (ov >= 0.35 and sim >= TRACK_THRESHOLD):
            return sim + 0.25 * ov
        return None

    def commit(self, f: Face, idx: int) -> np.ndarray:
        """Record that `f` is this person in frame `idx`; returns smoothed landmarks."""
        if not self._recent(idx):
            self.filt = None
        self.last_bbox, self.last_idx = f.bbox, idx
        if self.smoothing <= 0:
            return f.kps
        if self.filt is None:
            # beta scales with face size so fast moves stay locked on
            self.filt = OneEuro(self.smoothing, beta=6.0 / max(f.size, 16.0))
        return self.filt(f.kps.astype(np.float64), 1.0 / self.fps).astype(np.float32)


class FaceEngine:
    def __init__(self, model_dir: str = MODEL_DIR, providers: list[str] | None = None):
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        avail = ort.get_available_providers()
        if providers is None:
            providers = []
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            elif "CoreMLExecutionProvider" in avail and os.environ.get("LEADROLE_COREML", "1") != "0":
                # Apple silicon: run on the GPU / Neural Engine. Ops CoreML can't take fall back to CPU.
                providers.append(("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}))
            providers.append("CPUExecutionProvider")
        self.providers = providers
        names = [p if isinstance(p, str) else p[0] for p in providers]
        self.gpu = "CUDAExecutionProvider" in names
        self.model_dir = model_dir
        ctx = 0 if self.gpu else -1

        buffalo = os.path.join(model_dir, "buffalo_l")
        if "CoreMLExecutionProvider" in names:
            # Older ONNX Runtime builds don't take CoreML options; fall back to the plain provider.
            try:
                ort.InferenceSession(os.path.join(buffalo, "w600k_r50.onnx"), providers=providers)
            except Exception:  # noqa: BLE001
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                self.providers = providers
        self.det = get_model(os.path.join(buffalo, "det_10g.onnx"), providers=providers)
        self.det.prepare(ctx_id=ctx, input_size=(640, 640), det_thresh=0.5)
        self.rec = get_model(os.path.join(buffalo, "w600k_r50.onnx"), providers=providers)
        self.rec.prepare(ctx_id=ctx)
        swapper = get_model(os.path.join(model_dir, "inswapper_128.onnx"), providers=providers)
        self.swap_sess = swapper.session
        self.swap_in = [i.name for i in self.swap_sess.get_inputs()]
        self.swap_out = self.swap_sess.get_outputs()[0].name
        self.emap = swapper.emap.astype(np.float32)
        # ONNX Runtime silently falls back to CPU if an accelerator can't load - report what we really got
        got = self.swap_sess.get_providers()
        self.gpu = "CUDAExecutionProvider" in got
        self.coreml = "CoreMLExecutionProvider" in got
        self.accelerated = self.gpu or self.coreml
        self.device = "GPU (CUDA)" if self.gpu else "Apple silicon (CoreML)" if self.coreml else "CPU"
        if "CUDAExecutionProvider" in names and not self.gpu:
            print("WARNING: CUDA was requested but failed to load; running on CPU (much slower).", flush=True)
        if "CoreMLExecutionProvider" in names and not self.coreml:
            print("WARNING: CoreML was requested but failed to load; running on CPU.", flush=True)
        self._ort = ort
        self._lazy_lock = threading.Lock()
        self._gfpgan = None
        self._parser = None
        self._masks: dict[int, np.ndarray] = {}

    # ---------- lazy models (only loaded for tiers that need them) ----------
    def _session(self, name):
        return self._ort.InferenceSession(os.path.join(self.model_dir, name), providers=self.providers)

    @property
    def gfpgan(self):
        with self._lazy_lock:
            if self._gfpgan is None:
                self._gfpgan = self._session("gfpgan_1.4.onnx")
        return self._gfpgan

    @property
    def parser(self):
        with self._lazy_lock:
            if self._parser is None:
                self._parser = self._session("bisenet_resnet_34.onnx")
        return self._parser

    def warm(self, tier: Tier) -> None:
        if tier.enhance > 0:
            _ = self.gfpgan
        if tier.occlusion_mask:
            _ = self.parser

    def mask(self, size: int) -> np.ndarray:
        if size not in self._masks:
            self._masks[size] = box_mask(size)
        return self._masks[size]

    # ---------- analysis ----------
    def detect(self, img: np.ndarray, det_size: int = 640, embed: bool = True, min_size: int = 20) -> list[Face]:
        bboxes, kpss = self.det.detect(img, input_size=(det_size, det_size), max_num=0)
        faces = [Face(b[:4].astype(np.float32), k.astype(np.float32), float(b[4]))
                 for b, k in zip(bboxes, kpss)]
        faces = [f for f in faces if f.size >= min_size]
        if embed and faces:
            self.embed(img, faces)
        return faces

    def embed(self, img: np.ndarray, faces: list[Face]) -> None:
        """Compute identity embeddings for the faces that don't have one yet (the slow part on CPU)."""
        from insightface.utils import face_align
        todo = [f for f in faces if f.emb is None]
        if not todo:
            return
        crops = [face_align.norm_crop(img, landmark=f.kps, image_size=112) for f in todo]
        feats = self.rec.get_feat(crops)
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        for f, e in zip(todo, feats):
            f.emb = e.astype(np.float32)

    def detect_source(self, img: np.ndarray) -> tuple[Face, np.ndarray] | tuple[None, None]:
        """Largest face in a user photo (and the canvas its coordinates refer to).
        Tightly cropped selfies often defeat the detector, so retry with a padded
        canvas before giving up."""
        h, w = img.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for pad in (0, 0.5, 1.0):
            canvas = img
            if pad:
                p = int(max(img.shape[:2]) * pad)
                canvas = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.detect(canvas, 640, embed=True, min_size=12)
            if faces:
                return max(faces, key=lambda f: f.size * f.score), canvas
        return None, None

    def source_latent(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Average several photos of the same person into one identity code."""
        e = np.mean(np.stack(embeddings), axis=0)
        e = e / np.linalg.norm(e)
        latent = e.reshape(1, -1) @ self.emap
        return (latent / np.linalg.norm(latent)).astype(np.float32)

    # ---------- swap ----------
    def _swap_crop(self, frame, kps, latent, boost):
        from insightface.utils import face_align
        size = 128 * boost
        M = face_align.estimate_norm(kps, size)
        crop = cv2.warpAffine(frame, M, (size, size), borderValue=0.0)
        if boost == 1:
            tiles = crop[None]
        else:  # pixel boost: interleaved sub-pixel tiles, each swapped at 128px
            tiles = crop.reshape(128, boost, 128, boost, 3).transpose(1, 3, 0, 2, 4).reshape(boost * boost, 128, 128, 3)
        outs = []
        for t in tiles:
            blob = np.ascontiguousarray(t[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            pred = self.swap_sess.run([self.swap_out], {self.swap_in[0]: blob, self.swap_in[1]: latent})[0][0]
            outs.append(pred.transpose(1, 2, 0))
        if boost == 1:
            res = outs[0]
        else:
            res = np.stack(outs).reshape(boost, boost, 128, 128, 3).transpose(2, 0, 3, 1, 4).reshape(size, size, 3)
        res = np.clip(res * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, :, ::-1]
        return res, M

    def _restore(self, crop: np.ndarray) -> np.ndarray:
        x = crop[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        y = self.gfpgan.run(None, {"input": x})[0][0]
        y = (np.clip(y, -1, 1) + 1.0) * 127.5
        return np.clip(y.transpose(1, 2, 0) + 0.5, 0, 255).astype(np.uint8)[:, :, ::-1]

    def _face_region(self, crop: np.ndarray) -> np.ndarray:
        x = (crop[:, :, ::-1].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        logits = self.parser.run(None, {"input": x})[0][0]
        region = np.isin(logits.argmax(0), FACE_REGIONS).astype(np.float32)
        region = cv2.GaussianBlur(region, (0, 0), 5)
        return (region.clip(0.5, 1.0) - 0.5) * 2.0

    def swap_face(self, frame: np.ndarray, kps: np.ndarray, latent: np.ndarray, tier: Tier) -> np.ndarray:
        """Swap one face into `frame` (modified in place and returned)."""
        fake, M = self._swap_crop(frame, kps, latent, tier.pixel_boost)
        fake_mask = self.mask(fake.shape[0])
        M512 = None
        if tier.enhance > 0 or tier.occlusion_mask:
            M512 = cv2.estimateAffinePartial2D(kps.astype(np.float32), FFHQ_512, method=cv2.LMEDS)[0]
        if M512 is None:  # Fast tier: paste the swap straight back
            paste_roi(frame, fake, M, fake_mask)
            return frame

        # Work in a 512px face-aligned crop, then touch the frame exactly once.
        orig = cv2.warpAffine(frame, M512, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        crop = orig.copy()
        to3 = lambda a: np.vstack([a, [0.0, 0.0, 1.0]])  # noqa: E731
        M_rel = (to3(M) @ np.linalg.inv(to3(M512)))[:2]  # crop512 coords -> swap-crop coords
        paste_roi(crop, fake, M_rel, fake_mask)
        if tier.enhance > 0:
            restored = self._restore(crop)
            restored = cv2.addWeighted(restored, tier.enhance, crop, 1.0 - tier.enhance, 0)
            crop = match_low_frequencies(restored, crop)
        mask = self.mask(512)
        if tier.occlusion_mask:
            mask = np.minimum(mask, self._face_region(orig))
        paste_roi(frame, crop, M512, mask)
        return frame


@dataclass
class Role:
    """One face from the user's photos and who it replaces in the video."""
    latent: np.ndarray              # identity to apply
    target: np.ndarray | None       # identity of the person to replace (mode "identity")
    mode: str = "identity"          # "identity" | "largest" (biggest face in frame) | "all" (every face)


class FrameSwapper:
    """Per-render state: who replaces whom, and the tier.

    `analyze` is stateless (safe to run on many frames in parallel);
    `apply` must be called in frame order because tracking/smoothing carry over.
    """

    def __init__(self, engine: FaceEngine, tier: Tier, roles: list[Role], fps: float, vr_sbs: bool = False):
        if not roles:
            raise ValueError("Pick at least one person to replace.")
        if len(roles) > 1 and any(r.mode == "all" for r in roles):
            raise ValueError("'Everyone' can only be used with a single face.")
        self.engine, self.tier, self.roles, self.vr = engine, tier, roles, vr_sbs
        self.every_face = roles[0].mode == "all"
        self.needs_identity = any(r.mode == "identity" for r in roles)
        halves = 2 if vr_sbs else 1
        # one track per role (identity roles match by face; "largest" roles only use it for smoothing)
        self.tracks = [[TargetTrack(r.target, tier.smoothing, fps) for r in roles] if not self.every_face else []
                       for _ in range(halves)]
        engine.warm(tier)

    def analyze(self, frame: np.ndarray, embed: bool | None = None) -> list[list[Face]]:
        """Detect faces. With embed=True (GPU prefetch) identities are computed up front;
        otherwise they are computed lazily in `apply`, only when tracking needs them."""
        if embed is None:
            embed = self.needs_identity and self.engine.accelerated
        if not self.vr:
            return [self.engine.detect(frame, self.tier.det_size, embed=embed)]
        half = frame.shape[1] // 2
        return [self.engine.detect(np.ascontiguousarray(frame[:, :half]), self.tier.det_size, embed=embed),
                [f.offset(half) for f in
                 self.engine.detect(np.ascontiguousarray(frame[:, half:]), self.tier.det_size, embed=embed)]]

    VERIFY_EVERY = 12  # frames between identity re-checks on a locked track

    def _assign(self, frame, faces, tracks, idx) -> list[tuple[int, int]]:
        """(role, face index) pairs for this frame."""
        if not faces:
            return []
        ident = [r for r, role in enumerate(self.roles) if role.mode == "identity"]
        out: list[tuple[int, int]] = []
        if ident:
            out = self._assign_identity(frame, faces, tracks, idx, ident)
        # "Biggest face" roles take the largest faces nobody else claimed, in role order.
        taken = {i for _, i in out}
        free = sorted((i for i in range(len(faces)) if i not in taken), key=lambda i: -faces[i].size)
        for r, role in enumerate(self.roles):
            if role.mode == "largest" and free:
                out.append((r, free.pop(0)))
        return out

    def _assign_identity(self, frame, faces, tracks, idx, ident: list[int]) -> list[tuple[int, int]]:
        # Cheap path: every person we follow has exactly one face continuing its track.
        locks = {r: tracks[r].locked(faces, idx) for r in ident}
        if all(l is not None for l in locks.values()) and len(set(locks.values())) == len(locks):
            due = [r for r in ident if idx - tracks[r].verified_idx >= self.VERIFY_EVERY]
            if due:
                self.engine.embed(frame, [faces[locks[r]] for r in due])
            if all(tracks[r].score(faces[locks[r]], idx) is not None for r in due):
                for r in due:
                    tracks[r].verified_idx = idx
                return list(locks.items())
        # Full path: identify everyone, then assign jointly (best match first, each face and
        # person used once) so two of your faces can never land on the same person.
        self.engine.embed(frame, faces)
        cands = sorted(((s, r, i) for r in ident for i, f in enumerate(faces)
                        if (s := tracks[r].score(f, idx)) is not None), reverse=True)
        used_r, used_f, out = set(), set(), []
        for _, r, i in cands:
            if r in used_r or i in used_f:
                continue
            used_r.add(r)
            used_f.add(i)
            tracks[r].verified_idx = idx
            out.append((r, i))
        return out

    def focus(self, analysis: list[list[Face]], idx: int) -> np.ndarray | None:
        """Where the camera should look this frame: the first face's person (or the largest face)."""
        faces = analysis[0] if analysis else []
        if self.every_face:
            return max(faces, key=lambda f: f.size).bbox if faces else None
        tr = self.tracks[0][0]
        return tr.last_bbox if tr.last_idx == idx else None

    def track_only(self, frame: np.ndarray, analysis: list[list[Face]], idx: int) -> None:
        """Update tracking without swapping (used to warm up before a chunk starts)."""
        if self.every_face:
            return
        for half, faces in enumerate(analysis):
            tracks = self.tracks[half]
            for r, i in self._assign(frame, faces, tracks, idx):
                tracks[r].commit(faces[i], idx)

    def apply(self, frame: np.ndarray, analysis: list[list[Face]], idx: int) -> tuple[np.ndarray, list[int]]:
        """Swap the frame in place. Returns the role index of every face swapped."""
        swapped: list[int] = []
        for half, faces in enumerate(analysis):
            if self.every_face:
                for f in faces:
                    self.engine.swap_face(frame, f.kps, self.roles[0].latent, self.tier)
                    swapped.append(0)
                continue
            tracks = self.tracks[half]
            for r, i in self._assign(frame, faces, tracks, idx):
                kps = tracks[r].commit(faces[i], idx)
                self.engine.swap_face(frame, kps, self.roles[r].latent, self.tier)
                swapped.append(r)
        return frame, swapped
