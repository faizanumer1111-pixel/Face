"""Time ranges of the video to swap, in exact frame numbers.

A segment covers frames [start, end): `start` is the first frame swapped and `end`
is one past the last. At 30 fps, the segment 2.0s-5.5s is frames 60..164.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction

MODES = ("cast", "everyone", "main")
MODE_LABELS = {"cast": "Picked people", "everyone": "Everyone", "main": "Whoever's on screen"}
DEFAULT_THRESHOLD = 0.35


class SegmentError(ValueError):
    """A problem with the segments, worded for the person using the app."""


@dataclass
class Segment:
    start: int
    end: int
    mode: str = "everyone"
    face: str | None = None  # face to put in (everyone / main); None = the project's default face
    cast: list[dict] = field(default_factory=list)  # picked people: [{"ref": person id, "face": face id}]
    threshold: float = DEFAULT_THRESHOLD  # match strictness for picked people
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def length(self) -> int:
        return self.end - self.start

    def settings_key(self) -> tuple:
        if self.mode == "cast":
            return (self.mode, tuple((c["ref"], c.get("face")) for c in self.cast), round(self.threshold, 3))
        return (self.mode, self.face)

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Timecodes
# --------------------------------------------------------------------------- #


def frame_to_seconds(frame: int, fps: Fraction) -> float:
    return float(Fraction(frame) / fps)


def seconds_to_frame(seconds: float, fps: Fraction) -> int:
    """Nearest frame boundary to a time (used for segment edges)."""
    return int(math.floor(float(Fraction(seconds).limit_denominator(10**6) * fps) + 0.5))


def format_timecode(seconds: float) -> str:
    """M:SS.mmm, or H:MM:SS.mmm past an hour."""
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m}:{s:02d}.{ms:03d}"


_TC = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d*)?|\.\d+)$")


def parse_time(text: str | int | float, fps: Fraction) -> int:
    """Parse a timecode into a frame boundary.

    Accepts seconds ("2.5"), M:SS.mmm ("1:02.5"), H:MM:SS.mmm ("0:01:02.5") and frame
    numbers ("#120" or "f120"). Ints are treated as frame numbers.
    """
    if isinstance(text, bool):
        raise SegmentError(f"Couldn't read the time {text!r}.")
    if isinstance(text, int):
        return text
    if isinstance(text, float):
        return seconds_to_frame(text, fps)
    t = str(text).strip().lower().replace(",", ".")
    if t.startswith(("#", "f")) and t[1:].isdigit():
        return int(t[1:])
    m = _TC.match(t)
    if not m:
        raise SegmentError(f"Couldn't read the time \"{text}\". Use seconds (2.5), minutes:seconds (1:02.5) or a frame number (#120).")
    a, b, sec = m.groups()
    if a is not None and b is not None:
        hours, minutes = int(a), int(b)
    elif a is not None:
        hours, minutes = 0, int(a)
    else:
        hours, minutes = 0, 0
    total = hours * 3600 + minutes * 60 + float(sec)
    return seconds_to_frame(total, fps)


# --------------------------------------------------------------------------- #
# Normalising
# --------------------------------------------------------------------------- #


def from_json(items: list[dict], fps: Fraction) -> list[Segment]:
    segs = []
    for it in items:
        mode = it.get("mode", "everyone")
        cast = [dict(ref=str(c["ref"]), face=c.get("face") or None) for c in it.get("cast") or [] if c.get("ref")]
        if mode == "one":  # older single-person form
            mode = "cast"
            if it.get("ref"):
                cast = [{"ref": it["ref"], "face": it.get("face") or None}]
        if mode not in MODES:
            raise SegmentError(f"Unknown mode \"{mode}\". Use one of: {', '.join(MODES)}.")
        threshold = float(it.get("threshold", DEFAULT_THRESHOLD))
        if not 0.0 < threshold < 1.0:
            raise SegmentError("Match strictness has to be between 0 and 1.")
        seg = Segment(
            start=parse_time(it["start"], fps),
            end=parse_time(it["end"], fps),
            mode=mode,
            face=it.get("face") or None,
            cast=cast,
            threshold=threshold,
        )
        if it.get("id"):
            seg.id = str(it["id"])
        segs.append(seg)
    return segs


def normalize(segments: list[Segment], total_frames: int) -> list[Segment]:
    """Clamp to the video, sort, and merge overlapping segments that share settings.

    Overlapping segments with different settings are an error, because a frame can
    only be swapped one way.
    """
    clamped = []
    for s in segments:
        start = max(0, min(int(s.start), total_frames))
        end = max(0, min(int(s.end), total_frames))
        if end <= start:
            continue
        cast = [dict(c) for c in s.cast] if s.mode == "cast" else []
        face = None if s.mode == "cast" else s.face
        clamped.append(Segment(start, end, s.mode, face, cast, s.threshold, s.id))
    clamped.sort(key=lambda s: (s.start, s.end))
    out: list[Segment] = []
    for s in clamped:
        if out:
            prev = out[-1]
            if s.start <= prev.end and s.settings_key() == prev.settings_key():
                prev.end = max(prev.end, s.end)
                continue
            if s.start < prev.end:
                raise SegmentError(
                    f"Two segments overlap but use different settings "
                    f"(frames {prev.start}-{prev.end - 1} and {s.start}-{s.end - 1}). "
                    "Move one so they don't overlap, or give them the same mode."
                )
        out.append(s)
    return out


def stats(segments: list[Segment], total_frames: int) -> dict:
    process = sum(s.length for s in segments)
    skipped = (1 - process / total_frames) * 100 if total_frames else 0.0
    return {"frames_to_process": process, "total_frames": total_frames, "percent_skipped": round(skipped, 1)}


def check_ready(segments: list[Segment], references: set[str], faces: set[str], default_face: str | None) -> None:
    """Raise if the segments can't be rendered yet."""
    if not segments:
        raise SegmentError("Mark at least one segment first: press I at the start and O at the end.")
    for i, s in enumerate(segments, 1):
        if s.mode == "cast":
            if not s.cast:
                raise SegmentError(
                    f"Segment {i} is set to Picked people, but nobody is picked yet. "
                    "Pause inside it and drag a face from the faces panel onto a person."
                )
            for c in s.cast:
                if c["ref"] not in references:
                    raise SegmentError(f"Segment {i}: one of the picked people is missing. Pick them again.")
                if (c.get("face") or default_face) not in faces:
                    raise SegmentError(f"Segment {i}: the face for one of the picked people was deleted. Drag another face onto them.")
        elif (s.face or default_face) not in faces:
            raise SegmentError("Add a face to put in first: drop a photo onto the faces panel.")
