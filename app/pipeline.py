"""Rendering: decode, swap faces inside the segments only, encode.

Frames outside the segments are copied straight from the decoder to the encoder
and never go near a model. Decoding, swapping and encoding run on separate
threads so the encoder isn't left waiting.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .faces import Face, SwapOptions
from .processor import FrameProcessor, selector_for
from .segments import Segment
from .targets import make_selector
from .video_io import FrameReader, FrameWriter, VideoError, VideoInfo, finalize_vr
from .vr import Layout

log = logging.getLogger(__name__)

OUTPUT_MODES = ("full", "clips")


class Cancelled(Exception):
    pass


@dataclass
class RenderRequest:
    video_path: Path
    info: VideoInfo
    faces: dict[str, np.ndarray]  # face library: id -> embedding of the face to put in
    default_face: str | None
    segments: list[Segment]
    references: dict[str, np.ndarray]  # picked people: id -> embedding
    output_path: Path
    output_mode: str = "full"
    det_size: int = 640
    max_quality: bool = False
    options: SwapOptions = field(default_factory=SwapOptions)
    steady: bool = True
    vr_format: str = "flat"


@dataclass
class RenderStats:
    frames_written: int = 0
    frames_total: int = 0
    frames_swapped: int = 0  # frames sent to the face models
    swap_total: int = 0
    faces_swapped: int = 0
    swap_fps: float = 0.0
    eta_seconds: float | None = None
    elapsed: float = 0.0
    frames_without_target: int = 0
    breakdown: dict = field(default_factory=dict)  # milliseconds per swapped frame, by step

    def to_json(self) -> dict:
        out = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        out["breakdown"] = {k: round(v, 1) for k, v in self.breakdown.items()}
        return out


@dataclass
class _Timing:
    swap_t: float = 0.0
    pass_t: float = 0.0
    swap_n: int = 0
    pass_n: int = 0
    started: float = field(default_factory=time.time)


def process_frame(engine, frame: np.ndarray, selector, latent: np.ndarray, det_size: int,
                  opts: SwapOptions | None = None) -> tuple[list[Face], list[Face]]:
    """Detect, choose and swap faces in one normal frame (in place), all with one face."""
    faces = engine.detect(frame, det_size)
    if faces and selector.needs_embeddings:
        for f in faces:
            engine.embed(frame, f)
    targets = selector.select(faces, frame.shape)
    for f in targets:
        engine.swap(frame, f, latent, opts)
    return faces, targets


def latents_for(engine, faces: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {fid: engine.latent(emb) for fid, emb in faces.items()}


def render(req: RenderRequest, engine, on_progress: Callable[[RenderStats], None] | None = None,
           cancel: threading.Event | None = None, log_path: Path | None = None) -> RenderStats:
    if req.output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output mode {req.output_mode!r}")
    if not req.segments:
        raise ValueError("No segments to render")
    info = req.info
    fps = float(info.fps)
    cancel = cancel or threading.Event()
    latents = latents_for(engine, req.faces)
    layout = Layout.of(req.vr_format)
    segments = sorted(req.segments, key=lambda s: s.start)

    stats = RenderStats(swap_total=sum(s.length for s in segments))
    if req.output_mode == "full":
        stats.frames_total = info.total_frames
        ranges = [(0, info.total_frames)]
        audio = "full"
    else:
        stats.frames_total = stats.swap_total
        ranges = [(s.start, s.end) for s in segments]
        audio = ranges

    tmp = req.output_path.with_name(req.output_path.stem + ".partial.mp4")
    log_path = log_path or req.output_path.with_suffix(".log")
    writer = FrameWriter(tmp, info, req.video_path, audio, req.max_quality, log_path, vr=layout.is_vr,
                         index_at_end=not layout.is_flat_mono)
    # Keep about 400 MB of frames in flight at most (8K VR frames are ~90 MB each).
    depth = max(2, min(8, int(400e6 // max(info.frame_bytes, 1))))
    timing = _Timing()
    last_report = [0.0]
    engine_start = dict(getattr(engine, "timings", {}))
    waits = {"reading video": 0.0, "saving video": 0.0}

    def report(force: bool = False) -> None:
        now = time.time()
        if not force and now - last_report[0] < 0.25:
            return
        last_report[0] = now
        stats.elapsed = now - timing.started
        if timing.swap_n:
            stats.swap_fps = timing.swap_n / timing.swap_t if timing.swap_t > 0 else 0.0
        per_swap = timing.swap_t / timing.swap_n if timing.swap_n else 0.25
        per_pass = timing.pass_t / timing.pass_n if timing.pass_n else 0.002
        rem_swap = stats.swap_total - stats.frames_swapped
        rem_pass = (stats.frames_total - stats.frames_written) - rem_swap
        stats.eta_seconds = max(0.0, rem_swap * per_swap + max(0, rem_pass) * per_pass)
        if on_progress:
            on_progress(stats)

    # Writer thread: encode while the next frames are being swapped.
    out_q: queue.Queue = queue.Queue(maxsize=depth)
    writer_error: list[BaseException] = []

    def write_loop() -> None:
        try:
            while True:
                item = out_q.get()
                if item is None:
                    return
                writer.write(item)
        except BaseException as exc:  # surfaced after the loop
            writer_error.append(exc)
            while out_q.get() is not None:  # drain so the producer never blocks
                pass

    wthread = threading.Thread(target=write_loop, name="encoder", daemon=True)
    wthread.start()

    def put(frame: np.ndarray) -> None:
        if writer_error:
            raise writer_error[0]
        t0 = time.perf_counter()
        out_q.put(frame)
        waits["saving video"] += time.perf_counter() - t0

    try:
        for r_start, r_end in ranges:
            reader = FrameReader(req.video_path, info, start=r_start, count=r_end - r_start)
            in_q: queue.Queue = queue.Queue(maxsize=max(2, depth // 2))
            stop = threading.Event()

            def read_loop(reader=reader, n=r_end - r_start, in_q=in_q, stop=stop) -> None:
                # Keeps ffmpeg decoding while the main thread runs the models.
                try:
                    for _ in range(n):
                        if stop.is_set():
                            return
                        frame = reader.read()
                        if frame is None:
                            return
                        in_q.put(frame)
                finally:
                    in_q.put(None)

            rthread = threading.Thread(target=read_loop, name="decoder", daemon=True)
            rthread.start()
            try:
                seg_i = 0
                selector = None
                for offset in range(r_end - r_start):
                    if cancel.is_set():
                        raise Cancelled()
                    t_wait = time.perf_counter()
                    frame = in_q.get()
                    waits["reading video"] += time.perf_counter() - t_wait
                    if frame is None:
                        err = reader.error_text()
                        raise VideoError(f"Video ended early at frame {r_start + offset}. {err}".strip())
                    i = r_start + offset
                    while seg_i < len(segments) and segments[seg_i].end <= i:
                        seg_i += 1
                        selector = None
                    seg = segments[seg_i] if seg_i < len(segments) and segments[seg_i].start <= i else None
                    t0 = time.perf_counter()
                    if seg is not None:
                        if selector is None:
                            selector = FrameProcessor(engine, layout, latents, req.options, req.det_size,
                                                      selector_for(seg, req.references, req.default_face, fps),
                                                      steady=req.steady)
                        result = selector.process(frame)
                        stats.frames_swapped += 1
                        stats.faces_swapped += result.swapped
                        if not result.swapped:
                            stats.frames_without_target += 1
                        put(frame)
                        timing.swap_t += time.perf_counter() - t0
                        timing.swap_n += 1
                    else:
                        put(frame)
                        timing.pass_t += time.perf_counter() - t0
                        timing.pass_n += 1
                    stats.frames_written += 1
                    report()
            finally:
                stop.set()
                if reader.proc.poll() is None:
                    reader.proc.kill()  # unblocks the decoder thread if we stopped early
                while rthread.is_alive():
                    try:
                        in_q.get(timeout=0.1)
                    except queue.Empty:
                        pass
                reader.close()
        out_q.put(None)
        wthread.join()
        if writer_error:
            raise writer_error[0]
        writer.finish()
        if layout.is_vr or not layout.is_flat_mono:
            finalize_vr(tmp, req.output_path, req.vr_format)
        else:
            tmp.replace(req.output_path)
    except BaseException:
        if wthread.is_alive():
            # Unblock the writer thread, then kill the encoder.
            try:
                while True:
                    out_q.get_nowait()
            except queue.Empty:
                pass
            out_q.put(None)
            wthread.join(timeout=5)
        writer.abort()
        tmp.unlink(missing_ok=True)
        raise
    # Where the time went, per swapped frame (waits are spread over the whole render).
    n = max(1, stats.frames_swapped)
    for step, total in getattr(engine, "timings", {}).items():
        spent = total - engine_start.get(step, 0.0)
        if spent > 0:
            stats.breakdown[step] = spent * 1000 / n
    for step, total in waits.items():
        if total > 0.05:
            stats.breakdown[step] = total * 1000 / n
    report(force=True)
    stats.eta_seconds = 0.0
    return stats
