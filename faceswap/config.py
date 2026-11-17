"""Render tiers and model locations.

Every knob that trades speed for quality lives here so the three tiers are easy
to tune without touching the pipeline code.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

MODEL_DIR = os.environ.get("FACESWAP_MODEL_DIR", "/models")

# All models come from GitHub release mirrors (no Hugging Face dependency).
MODEL_URLS = {
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    "bisenet_resnet_34.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx",
    # det_10g.onnx + w600k_r50.onnx are extracted from this pack
    "buffalo_l.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}


@dataclass(frozen=True)
class Tier:
    key: str
    label: str
    blurb: str
    det_size: int           # face detector input (bigger finds smaller faces, slower)
    pixel_boost: int        # 1 -> 128px swap, 2 -> 256px swap (4 sub-pixel passes)
    enhance: float          # GFPGAN blend (0 = off)
    occlusion_mask: bool    # face-parsing mask so hands/mics/hair stay in front of the face
    smoothing: float        # One-Euro min cutoff in Hz for landmark smoothing (0 = off)
    crf: int                # x264 quality (lower = better/larger)
    preset: str             # x264 speed preset
    max_chunks: int         # max GPUs rendering one video in parallel
    min_chunk_seconds: float
    speed: int              # 1-3, shown in the UI
    quality: int            # 1-3, shown in the UI

    def public(self) -> dict:
        return asdict(self)


TIERS: dict[str, Tier] = {
    "fast": Tier(
        key="fast", label="Fast",
        blurb="Straight swap at 128px. Quickest turnaround for drafts and approvals.",
        det_size=480, pixel_boost=1, enhance=0.0, occlusion_mask=False, smoothing=0.0,
        crf=20, preset="veryfast", max_chunks=12, min_chunk_seconds=2.5, speed=3, quality=1,
    ),
    "better": Tier(
        key="better", label="Better",
        blurb="Adds face restoration and steadier tracking for sharper skin, eyes and teeth.",
        det_size=640, pixel_boost=1, enhance=0.7, occlusion_mask=False, smoothing=1.5,
        crf=17, preset="medium", max_chunks=10, min_chunk_seconds=3.0, speed=2, quality=2,
    ),
    "premium": Tier(
        key="premium", label="Premium",
        blurb="256px multi-pass swap, restoration, occlusion-aware blending and film-grade encode.",
        det_size=800, pixel_boost=2, enhance=0.85, occlusion_mask=True, smoothing=1.0,
        crf=14, preset="slow", max_chunks=8, min_chunk_seconds=3.0, speed=1, quality=3,
    ),
}


def get_tier(key: str) -> Tier:
    if key not in TIERS:
        raise ValueError(f"Unknown quality tier '{key}'. Pick one of: {', '.join(TIERS)}")
    return TIERS[key]


# Identity matching (ArcFace cosine similarity)
MATCH_THRESHOLD = 0.28      # face counts as the chosen person above this
TRACK_THRESHOLD = 0.12      # ...or above this when it overlaps where the person just was
CLUSTER_THRESHOLD = 0.42    # scan: faces above this join the same person
MAX_SOURCE_PHOTOS = 5
