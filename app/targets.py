"""Deciding which detected faces get swapped in each frame.

One selector is created per segment, so tracking starts fresh at every segment.
"""

from __future__ import annotations

import math

import numpy as np

from .faces import Face


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


class EveryoneSelector:
    """Swap every detected face."""

    needs_embeddings = False

    def __init__(self, source: str | None = None):
        self.source = source

    def select(self, faces: list[Face], frame_shape: tuple) -> list[Face]:
        for f in faces:
            f.source = self.source
        return list(faces)


class MainFaceSelector:
    """Swap only the main face: the largest, ties broken by closeness to the centre.

    Hysteresis keeps the swap from jumping between two faces of similar size: the
    current main face only loses its place once another face has been at least
    `switch_ratio` times bigger for `switch_frames` frames in a row. If the main
    face vanishes (a cut), a similarly big face takes over at once; a small
    background face only takes over after `max_gap` frames, so one missed
    detection doesn't swap a bystander.
    """

    needs_embeddings = False

    def __init__(self, switch_ratio: float = 1.3, switch_frames: int = 5, iou_match: float = 0.3,
                 tie_ratio: float = 0.9, takeover_ratio: float = 0.5, max_gap: int = 3, source: str | None = None):
        self.source = source
        self.switch_ratio = switch_ratio
        self.switch_frames = switch_frames
        self.iou_match = iou_match
        self.tie_ratio = tie_ratio
        self.takeover_ratio = takeover_ratio
        self.max_gap = max_gap
        self.current_bbox: np.ndarray | None = None
        self.current_area = 0.0
        self.streak = 0
        self.missing = 0

    def _best(self, faces: list[Face], frame_shape: tuple) -> Face:
        largest = max(f.area for f in faces)
        contenders = [f for f in faces if f.area >= largest * self.tie_ratio]
        cx, cy = frame_shape[1] / 2.0, frame_shape[0] / 2.0
        return min(contenders, key=lambda f: (math.hypot(f.center[0] - cx, f.center[1] - cy), -f.area))

    def _take(self, face: Face) -> list[Face]:
        face.source = self.source
        self.current_bbox = face.bbox.copy()
        self.current_area = face.area
        self.missing = 0
        return [face]

    def select(self, faces: list[Face], frame_shape: tuple) -> list[Face]:
        if not faces:
            self.missing += 1
            self.streak = 0
            if self.missing > self.max_gap:
                self.current_bbox = None
            return []
        if self.current_bbox is None:
            self.streak = 0
            return self._take(self._best(faces, frame_shape))

        overlaps = [iou(f.bbox, self.current_bbox) for f in faces]
        j = int(np.argmax(overlaps))
        if overlaps[j] >= self.iou_match:
            tracked = faces[j]
            others = [f for i, f in enumerate(faces) if i != j]
            challenger = max(others, key=lambda f: f.area) if others else None
            if challenger is not None and challenger.area >= self.switch_ratio * tracked.area:
                self.streak += 1
            else:
                self.streak = 0
            if challenger is not None and self.streak >= self.switch_frames:
                self.streak = 0
                return self._take(challenger)
            return self._take(tracked)

        # The main face isn't in this frame.
        self.missing += 1
        self.streak = 0
        candidate = self._best(faces, frame_shape)
        if candidate.area >= self.takeover_ratio * self.current_area or self.missing > self.max_gap:
            return self._take(candidate)
        return []


class ReferenceSelector:
    """Swap only the picked person, matched by face embedding.

    A face is a match when its cosine similarity to the reference is at least
    `threshold`. Once matched, the same face is followed by box overlap, so the swap
    holds through short dips in similarity (profile views, motion blur) for up to
    `max_dip_frames`, as long as similarity stays above `floor`. At most one face
    per frame is swapped.
    """

    needs_embeddings = True

    def __init__(self, reference: np.ndarray, threshold: float = 0.35, fps: float = 30.0,
                 floor: float = 0.15, iou_match: float = 0.3, dip_seconds: float = 2.0, max_missing: int = 5):
        ref = np.asarray(reference, dtype=np.float32).ravel()
        self.reference = ref / max(float(np.linalg.norm(ref)), 1e-12)
        self.threshold = threshold
        self.floor = floor
        self.iou_match = iou_match
        self.max_dip_frames = max(1, int(round(dip_seconds * fps)))
        self.max_missing = max_missing
        self.last_bbox: np.ndarray | None = None
        self.dip = 0
        self.missing = 0
        self.last_similarities: list[float] = []

    def similarity(self, face: Face) -> float:
        if face.embedding is None:
            raise ValueError("ReferenceSelector needs face embeddings")
        return float(np.dot(face.embedding, self.reference))

    def _lose(self) -> None:
        self.last_bbox = None
        self.dip = 0
        self.missing = 0

    def select(self, faces: list[Face], frame_shape: tuple) -> list[Face]:
        sims = [self.similarity(f) for f in faces]
        self.last_similarities = sims
        if not faces:
            self.missing += 1
            if self.missing > self.max_missing:
                self._lose()
            return []

        tracked = None
        if self.last_bbox is not None:
            overlaps = [iou(f.bbox, self.last_bbox) for f in faces]
            j = int(np.argmax(overlaps))
            if overlaps[j] >= self.iou_match and sims[j] >= self.floor:
                tracked = j

        strong = [i for i, s in enumerate(sims) if s >= self.threshold]
        if strong:
            pick = max(strong, key=lambda i: sims[i])
            self.dip = 0
        elif tracked is not None and self.dip < self.max_dip_frames:
            pick = tracked
            self.dip += 1
        else:
            self._lose()
            return []
        self.last_bbox = faces[pick].bbox.copy()
        self.missing = 0
        return [faces[pick]]


class CastSelector:
    """Each picked person gets their own face (the casting board).

    `cast` is a list of (reference embedding, face id). Every person is followed by
    their own ReferenceSelector; if two picked people claim the same face in a frame,
    the closer match wins.
    """

    needs_embeddings = True

    def __init__(self, cast: list[tuple[np.ndarray, str | None]], threshold: float = 0.35, fps: float = 30.0):
        self.subs = [(ReferenceSelector(emb, threshold=threshold, fps=fps), src) for emb, src in cast]

    def select(self, faces: list[Face], frame_shape: tuple) -> list[Face]:
        picks: dict[int, tuple[float, str | None, Face]] = {}
        for sel, src in self.subs:
            got = sel.select(faces, frame_shape)
            if got:
                f = got[0]
                sim = sel.similarity(f)
                if id(f) not in picks or sim > picks[id(f)][0]:
                    picks[id(f)] = (sim, src, f)
        out = []
        for _, src, f in picks.values():
            f.source = src
            out.append(f)
        return out


class KpsSmoother:
    """Steadies faces: smooths each face's landmarks from frame to frame.

    Small jitter is averaged away; real movement passes straight through, so the new
    face doesn't lag behind a moving head.
    """

    def __init__(self, iou_match: float = 0.3):
        self.iou_match = iou_match
        self.tracks: list[tuple[np.ndarray, np.ndarray]] = []

    def smooth(self, faces: list[Face]) -> None:
        new, used = [], set()
        for f in faces:
            best, best_iou = None, 0.0
            for i, (bb, _) in enumerate(self.tracks):
                if i not in used:
                    v = iou(f.bbox, bb)
                    if v > best_iou:
                        best, best_iou = i, v
            if best is not None and best_iou >= self.iou_match:
                used.add(best)
                prev = self.tracks[best][1]
                width = max(1.0, float(f.bbox[2] - f.bbox[0]))
                moved = float(np.linalg.norm(f.kps - prev, axis=1).mean()) / width
                a = min(1.0, 0.25 + moved * 8)
                f.kps = (a * f.kps + (1 - a) * prev).astype(np.float32)
            new.append((f.bbox.copy(), np.asarray(f.kps, np.float32).copy()))
        self.tracks = new


def make_selector(mode: str, reference: np.ndarray | None = None, threshold: float = 0.35, fps: float = 30.0,
                  source: str | None = None, cast: list[tuple[np.ndarray, str | None]] | None = None):
    if mode == "everyone":
        return EveryoneSelector(source)
    if mode == "main":
        return MainFaceSelector(source=source)
    if mode == "one":
        if reference is None:
            raise ValueError("One person mode needs a reference face")
        return CastSelector([(reference, source)], threshold=threshold, fps=fps)
    if mode == "cast":
        if not cast:
            raise ValueError("Picked people mode needs at least one person")
        return CastSelector(cast, threshold=threshold, fps=fps)
    raise ValueError(f"Unknown mode {mode!r}")
