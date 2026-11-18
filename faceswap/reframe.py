"""Reframe: crop the output to an ad format (9:16, 1:1, 4:5), following the chosen person.

The crop window is the full height (or width) of the source at the target aspect, so no pixels
are scaled, only cut away. In `follow` mode the window glides after the person being replaced
(a One-Euro filter keeps it from twitching); in `fixed` mode it stays where the user put it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import OneEuro

ASPECTS = {
    "original": None,
    "9:16": (9, 16),   # Reels, TikTok, Stories
    "4:5": (4, 5),     # Instagram / Facebook feed
    "1:1": (1, 1),     # square feed
    "16:9": (16, 9),   # landscape from a portrait source
}


def parse_frame(d: dict | None) -> dict:
    """{"aspect": "9:16", "mode": "follow"|"fixed", "cx": 0..1, "cy": 0..1}; aspect "original" = no crop."""
    d = d or {}
    aspect = str(d.get("aspect", "original"))
    if aspect not in ASPECTS:
        raise ValueError(f"Unknown format '{aspect}'. Pick one of: {', '.join(ASPECTS)}")
    mode = str(d.get("mode", "follow"))
    if mode not in ("follow", "fixed"):
        raise ValueError("Reframe mode must be 'follow' or 'fixed'.")
    cx, cy = float(d.get("cx", 0.5)), float(d.get("cy", 0.5))
    return {"aspect": aspect, "mode": mode, "cx": min(max(cx, 0.0), 1.0), "cy": min(max(cy, 0.0), 1.0)}


def crop_size(src_w: int, src_h: int, aspect: str) -> tuple[int, int]:
    """Output (width, height) for the aspect, cut from the source without scaling. Even numbers."""
    a = ASPECTS[aspect]
    if a is None:
        return src_w, src_h
    target = a[0] / a[1]
    if src_w / src_h > target:
        ch, cw = src_h, src_h * target
    else:
        cw, ch = src_w, src_w / target
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    return max(2, min(cw, src_w // 2 * 2)), max(2, min(ch, src_h // 2 * 2))


@dataclass
class Reframer:
    src_w: int
    src_h: int
    frame: dict
    fps: float

    def __post_init__(self):
        self.cw, self.ch = crop_size(self.src_w, self.src_h, self.frame["aspect"])
        self.active = self.frame["aspect"] != "original" and (self.cw, self.ch) != (self.src_w, self.src_h)
        self.follow = self.active and self.frame["mode"] == "follow"
        self.center = np.array([self.frame["cx"] * self.src_w, self.frame["cy"] * self.src_h], np.float64)
        # slow, steady camera move: low cutoff, a little speed-up for big moves
        self.filt = OneEuro(min_cutoff=0.4, beta=0.0008, d_cutoff=0.5) if self.follow else None
        self.seen = False

    def _clamp(self, c: np.ndarray) -> np.ndarray:
        return np.array([min(max(c[0], self.cw / 2), self.src_w - self.cw / 2),
                         min(max(c[1], self.ch / 2), self.src_h - self.ch / 2)])

    def observe(self, bbox: np.ndarray | None) -> None:
        """Feed the followed person's box for this frame (None when they're not on screen)."""
        if not self.follow:
            return
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face = np.array([(x1 + x2) / 2, (y1 + y2) / 2 - 0.15 * (y2 - y1)])  # a little headroom
            if not self.seen:
                self.filt.x = None  # start exactly on the person, no glide in from the centre
                self.seen = True
            self.center = self._clamp(self.filt(face, 1.0 / self.fps))
        elif self.seen:
            # hold the last position; run the filter so it stays continuous when they return
            self.center = self._clamp(self.filt(self.center, 1.0 / self.fps))

    def rect(self) -> tuple[int, int, int, int]:
        c = self._clamp(self.center)
        x0 = int(round(c[0] - self.cw / 2))
        y0 = int(round(c[1] - self.ch / 2))
        x0 = min(max(x0, 0), self.src_w - self.cw)
        y0 = min(max(y0, 0), self.src_h - self.ch)
        return x0, y0, self.cw, self.ch

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.active:
            return img
        x0, y0, w, h = self.rect()
        return np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])

    def public_rect(self) -> dict:
        x0, y0, w, h = self.rect()
        return {"x": x0 / self.src_w, "y": y0 / self.src_h, "w": w / self.src_w, "h": h / self.src_h,
                "width": w, "height": h}
