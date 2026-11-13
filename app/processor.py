"""Swapping the faces of one frame: normal video, 3D (two eyes) and VR (two eyes + stretch fix)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .faces import Face, SwapOptions
from .segments import Segment
from .targets import KpsSmoother, iou, make_selector
from .vr import Layout, eyes, match_between_eyes, view_for_face


@dataclass
class Plan:
    jobs: list  # (eye, face, latent) for every swap to do
    faces: int = 0
    similarities: list | None = None


@dataclass
class FrameResult:
    faces: int = 0  # faces found (left eye for 3D/VR)
    swapped: int = 0  # faces swapped (all eyes)
    similarities: list | None = None


def selector_for(seg: Segment | None, references: dict[str, np.ndarray], default_face: str | None, fps: float,
                 mode: str | None = None):
    if seg is None:
        return make_selector(mode or "everyone", source=default_face, fps=fps)
    if seg.mode == "cast":
        cast = [(references[c["ref"]], c.get("face") or default_face) for c in seg.cast if c["ref"] in references]
        return make_selector("cast", cast=cast, threshold=seg.threshold, fps=fps)
    return make_selector(seg.mode, source=seg.face or default_face, fps=fps)


class _ViewTrack:
    """Smooths a VR face's view direction between frames so the swap doesn't wobble."""

    def __init__(self):
        self.items: list[tuple[np.ndarray, tuple[float, float, float], KpsSmoother]] = []

    def get(self, bbox: np.ndarray, params: tuple[float, float, float], steady: bool):
        best, best_iou = None, 0.0
        for i, (bb, _, _) in enumerate(self.items):
            v = iou(bbox, bb)
            if v > best_iou:
                best, best_iou = i, v
        smoother = KpsSmoother()
        if best is not None and best_iou >= 0.3:
            _, prev, smoother = self.items.pop(best)
            if steady:
                moved = math.hypot(params[0] - prev[0], params[1] - prev[1]) / max(params[2], 1e-3)
                a = min(1.0, 0.4 + moved * 10)
                params = tuple(a * p + (1 - a) * q for p, q in zip(params, prev))
        self.items.append((bbox.copy(), params, smoother))
        return params, smoother


class FrameProcessor:
    """Detects, chooses and swaps faces in frames of one segment. Create one per segment."""

    def __init__(self, engine, layout: Layout, latents: dict[str | None, np.ndarray], opts: SwapOptions,
                 det_size: int, selector, steady: bool = True):
        self.engine = engine
        self.layout = layout
        self.latents = latents
        self.opts = opts
        self.det_size = det_size
        self.selector = selector
        self.steady = steady
        n_eyes = 1 if layout.stereo == "mono" else 2
        self.smoothers = [KpsSmoother() for _ in range(n_eyes)]
        self.view_tracks = [_ViewTrack() for _ in range(n_eyes)]

    def _latent(self, face: Face) -> np.ndarray | None:
        return self.latents.get(face.source)

    @property
    def can_split(self) -> bool:
        """Normal and 3D frames can be planned in order and swapped in parallel; VR can't (yet)."""
        return not self.layout.is_vr

    def process(self, frame: np.ndarray) -> FrameResult:
        if self.layout.is_vr:
            return self._process_vr(frame)
        return self.apply(frame, self.plan(frame))

    # ----------------------------------------------------------- normal and 3D
    def plan(self, frame: np.ndarray) -> Plan:
        """Find the faces and decide who gets which face. Must run frame by frame, in order,
        because tracking and smoothing carry over from one frame to the next."""
        views = eyes(frame, self.layout)
        found = [self.engine.detect(v, self.det_size) for v in views]
        left = found[0]
        if left and self.selector.needs_embeddings:
            for f in left:
                self.engine.embed(views[0], f)
        targets = self.selector.select(left, views[0].shape)
        per_eye = [(0, targets)]
        if len(views) == 2:
            pairs = match_between_eyes(left, found[1])
            chosen = {id(t) for t in targets}
            right_targets = []
            for i, f in enumerate(left):
                if id(f) in chosen and i in pairs:
                    pairs[i].source = f.source
                    right_targets.append(pairs[i])
            per_eye.append((1, right_targets))
        jobs = []
        for eye, faces in per_eye:
            if self.steady:
                self.smoothers[eye].smooth(faces)
            for f in faces:
                lat = self._latent(f)
                if lat is not None:
                    jobs.append((eye, f, lat))
        return Plan(jobs=jobs, faces=len(left), similarities=getattr(self.selector, "last_similarities", None))

    def apply(self, frame: np.ndarray, plan: Plan) -> FrameResult:
        """Do the planned swaps. Frames are independent here, so several can run at once."""
        views = eyes(frame, self.layout)
        for eye, f, lat in plan.jobs:
            self.engine.swap(views[eye], f, lat, self.opts)
        return FrameResult(faces=plan.faces, swapped=len(plan.jobs), similarities=plan.similarities)

    # ----------------------------------------------------------------- VR
    def _view_face(self, eye_img: np.ndarray, eye: int, face_eq: Face):
        """Render a flat view around a face, find the face in it and return (view, image, face)."""
        v = view_for_face(eye_img.shape, self.layout.projection, face_eq.bbox)
        (yaw, pitch, fov), smoother = self.view_tracks[eye].get(face_eq.bbox, (v.yaw, v.pitch, v.fov), self.steady)
        v = type(v)(eye_img.shape, self.layout.projection, yaw, pitch, fov, v.size)
        img = v.render(eye_img)
        faces = self.engine.detect(img, min(640, v.size))
        if not faces:
            return None
        c = v.size / 2
        vf = min(faces, key=lambda f: math.hypot(f.center[0] - c, f.center[1] - c))
        if self.steady:
            smoother.smooth([vf])
        return v, img, vf

    def _process_vr(self, frame: np.ndarray) -> FrameResult:
        eye_imgs = eyes(frame, self.layout)
        found = [self.engine.detect(e, self.det_size) for e in eye_imgs]
        left = found[0]
        rendered: dict[int, tuple] = {}
        if left and self.selector.needs_embeddings:
            for i, f in enumerate(left):
                r = self._view_face(eye_imgs[0], 0, f)
                if r is not None:
                    rendered[i] = r
                    f.embedding = self.engine.embed(r[1], r[2])
                else:
                    f.embedding = self.engine.embed(eye_imgs[0], f)
        targets = self.selector.select(left, eye_imgs[0].shape)
        res = FrameResult(faces=len(left), similarities=getattr(self.selector, "last_similarities", None))
        pairs = match_between_eyes(left, found[1]) if len(eye_imgs) == 2 else {}
        chosen = {id(t) for t in targets}
        for i, f in enumerate(left):
            if id(f) not in chosen:
                continue
            lat = self._latent(f)
            if lat is None:
                continue
            jobs = [(0, f, rendered.get(i))]
            if i in pairs:
                jobs.append((1, pairs[i], None))
            for eye, face_eq, r in jobs:
                r = r or self._view_face(eye_imgs[eye], eye, face_eq)
                if r is None:
                    continue
                view, img, vf = r
                edited = img.copy()
                self.engine.swap(edited, vf, lat, self.opts)
                view.paste(eye_imgs[eye], img, edited)
                res.swapped += 1
        return res
