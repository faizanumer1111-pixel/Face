"""The studio's operations, independent of where they run (local process or Modal GPUs).

Each function takes the engine/store it needs and returns JSON-friendly data.
User-facing problems raise ValueError with a message that can be shown as-is.
"""
from __future__ import annotations

import base64
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import CLUSTER_THRESHOLD, FACE_SLOTS, MAX_SOURCE_PHOTOS, get_tier
from .engine import FaceEngine, FrameSwapper, Role
from .jobs import JobStore
from .media import FrameReader, FrameWriter, concat_and_mux, grab_frame, jpeg


def parse_cast(d: dict) -> list[dict]:
    """Who replaces whom: [{"face": "a", "person": "p1"}, {"face": "b", "person": "p3"}].
    `person` may be "all" (every face) only when a single face is used."""
    cast = d.get("cast")
    if cast is None:  # single-face shorthand
        cast = [{"face": d.get("face", "a"), "person": d.get("person")}]
    if not isinstance(cast, list) or not cast:
        raise ValueError("Pick who to replace.")
    if len(cast) > len(FACE_SLOTS):
        raise ValueError(f"Up to {len(FACE_SLOTS)} faces can be swapped in one render.")
    out, faces, people = [], set(), set()
    for c in cast:
        face, person = str(c.get("face", "")), c.get("person") or None
        if face not in FACE_SLOTS or len(face) != 1:
            raise ValueError(f"Unknown face '{face}'.")
        label = f"Face {face.upper()}"
        if person is None:
            raise ValueError(f"Pick who {label} replaces.")
        if face in faces:
            raise ValueError(f"{label} is listed twice.")
        if person in people:
            raise ValueError("Two faces are set to replace the same person. Give each face its own person.")
        if person == "all" and len(cast) > 1:
            raise ValueError("'Every face' only works with a single face. Remove the other faces or pick people.")
        faces.add(face)
        people.add(person)
        out.append({"face": face, "person": person})
    return out


@dataclass
class RenderRequest:
    start: float
    end: float
    cast: list            # see parse_cast
    tier: str = "fast"
    vr: bool = False

    @classmethod
    def parse(cls, d: dict) -> "RenderRequest":
        get_tier(d.get("tier", "fast"))
        return cls(start=float(d.get("start", 0)), end=float(d.get("end", 0)), cast=parse_cast(d),
                   tier=d.get("tier", "fast"), vr=bool(d.get("vr", False)))


def data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def face_thumb(img: np.ndarray, bbox, size: int = 192, scale: float = 1.7) -> str:
    x1, y1, x2, y2 = bbox
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    k = size / s
    M = np.array([[k, 0, size / 2 - cx * k], [0, k, size / 2 - cy * k]], np.float32)
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(48, 46, 44))
    return data_url(jpeg(crop, quality=85))


# ---------------------------------------------------------------- the faces you bring
def analyze_faces(engine: FaceEngine, store: JobStore, jid: str, slot: str = "a") -> dict:
    """Check the photos for one face (A-D) and store its identity."""
    photos, embs = [], []
    for p in store.face_paths(jid, slot)[:MAX_SOURCE_PHOTOS]:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            photos.append({"name": p.name, "ok": False, "error": "Can't open this image. Use a JPG, PNG or WebP."})
            continue
        face, canvas = engine.detect_source(img)
        if face is None:
            photos.append({"name": p.name, "ok": False, "error": "No face found. Use a clear, front-facing photo."})
            continue
        photos.append({"name": p.name, "ok": True, "thumb": face_thumb(canvas, face.bbox, 160)})
        embs.append(face.emb)
    # Photos of someone else would blur the identity - drop the odd ones out.
    if len(embs) >= 2:
        mean = np.mean(embs, axis=0)
        mean /= np.linalg.norm(mean)
        ok_photos = [ph for ph in photos if ph["ok"]]
        keep = []
        for ph, e in zip(ok_photos, embs):
            if float(e @ mean) < 0.35:
                ph.update(ok=False, error="Looks like a different person from the other photos of this face, "
                                          "so it was left out.")
            else:
                keep.append(e)
        embs = keep or embs[:1]
    store.write_json(jid, f"source_{slot}.json", {"embeddings": [e.tolist() for e in embs]})
    return {"face": slot, "photos": photos, "usable": len(embs)}


def clear_face(store: JobStore, jid: str, slot: str) -> None:
    import shutil
    shutil.rmtree(store.face_dir(jid, slot), ignore_errors=True)
    (store.job_dir(jid) / f"source_{slot}.json").unlink(missing_ok=True)


def face_embeddings(store: JobStore, jid: str, slot: str) -> list:
    store.face_dir(jid, slot)  # validates the slot
    return store.read_json(jid, f"source_{slot}.json", {"embeddings": []})["embeddings"]


def load_identity(engine: FaceEngine, store: JobStore, jid: str, slot: str = "a") -> np.ndarray:
    embs = face_embeddings(store, jid, slot)
    if not embs:
        raise ValueError(f"Add a clear photo for Face {slot.upper()} first.")
    return engine.source_latent([np.asarray(e, np.float32) for e in embs])


def load_roles(engine: FaceEngine, store: JobStore, jid: str, cast: list[dict]) -> list[Role]:
    return [Role(load_identity(engine, store, jid, c["face"]), load_target(store, jid, c["person"])) for c in cast]


def load_target(store: JobStore, jid: str, person: str | None) -> np.ndarray | None:
    if person in (None, "", "all"):
        return None
    for p in store.read_json(jid, "people.json", {"people": []})["people"]:
        if p["id"] == person:
            return np.asarray(p["centroid"], np.float32)
    raise ValueError("Scan the video for people again, then pick who to replace.")


# ---------------------------------------------------------------- who's in the video
def scan_people(engine: FaceEngine, store: JobStore, jid: str, start: float, end: float,
                max_samples: int = 32) -> dict:
    src, meta = str(store.video_path(jid)), store.read_json(jid, "meta.json")
    start = max(0.0, min(start, meta["duration"]))
    end = min(meta["duration"], end) if end > start else meta["duration"]
    dur = max(end - start, 1.0 / meta["fps"])
    n = int(min(max_samples, max(6, dur * 3)))
    times = [start + dur * (i + 0.5) / n for i in range(n)]
    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(lambda t: grab_frame(src, meta, t), times))

    dets = []
    for i, frame in enumerate(frames):
        for f in engine.detect(frame, 640, embed=True, min_size=24):
            dets.append((i, f))
    dets.sort(key=lambda d: -d[1].size * d[1].score)  # strongest views anchor each person

    clusters: list[dict] = []
    for i, f in dets:
        best, best_sim = None, -1.0
        for c in clusters:
            sim = float(f.emb @ (c["sum"] / np.linalg.norm(c["sum"])))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= CLUSTER_THRESHOLD:
            best["sum"] += f.emb
            best["members"].append((i, f))
        else:
            clusters.append({"sum": f.emb.copy(), "members": [(i, f)]})

    people, stored = [], []
    for c in clusters:
        seen = sorted({i for i, _ in c["members"]})
        top_i, top_f = c["members"][0]
        if len(seen) == 1 and top_f.size < 48 and len(clusters) > 1:
            continue  # tiny one-off background faces
        centroid = c["sum"] / np.linalg.norm(c["sum"])
        people.append({
            "thumb": face_thumb(frames[top_i], top_f.bbox),
            "share": round(len(seen) / n, 3),
            "times": [round(times[i], 3) for i in seen],
            "face_px": int(np.median([f.size for _, f in c["members"]])),
        })
        stored.append(centroid.tolist())
    order = sorted(range(len(people)), key=lambda k: (-people[k]["share"], -people[k]["face_px"]))
    people = [dict(people[k], id=f"p{r}") for r, k in enumerate(order)]
    store.write_json(jid, "people.json", {"people": [{"id": f"p{r}", "centroid": stored[k]}
                                                     for r, k in enumerate(order)]})
    return {"people": people, "samples": n, "range": [start, end]}


# ---------------------------------------------------------------- single-frame preview
def preview(engine: FaceEngine, store: JobStore, jid: str, t: float, cast: list[dict],
            tier_key: str, vr: bool) -> dict:
    tier = get_tier(tier_key)
    roles = load_roles(engine, store, jid, cast)
    src, meta = str(store.video_path(jid)), store.read_json(jid, "meta.json")
    frame = grab_frame(src, meta, t)
    before = jpeg(frame)
    t0 = time.time()
    sw = FrameSwapper(engine, tier, roles, meta["fps"], vr)
    analysis = sw.analyze(frame)
    out, swapped = sw.apply(frame, analysis, 0)
    return {"before": data_url(before), "after": data_url(jpeg(out)), "t": t,
            "faces": sum(len(a) for a in analysis), "swapped": len(swapped),
            "found": [c["face"] for r, c in enumerate(cast) if r in swapped],
            "ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- full render
def plan_render(store: JobStore, jid: str, req: RenderRequest, max_chunks: int | None = None) -> dict:
    meta, tier = store.read_json(jid, "meta.json"), get_tier(req.tier)
    fps = meta["fps"]
    f0 = max(0, int(round(req.start * fps)))
    f1 = min(meta["frames"], int(round(req.end * fps))) if req.end > req.start else meta["frames"]
    total = f1 - f0
    if total < 1:
        raise ValueError("The selected section is empty. Drag the in and out handles apart.")
    limit = tier.max_chunks if max_chunks is None else min(max_chunks, tier.max_chunks)
    n = max(1, min(limit, math.ceil(total / (tier.min_chunk_seconds * fps))))
    bounds = [f0 + round(total * i / n) for i in range(n + 1)]
    chunks = [{"index": i, "start_frame": bounds[i], "count": bounds[i + 1] - bounds[i]} for i in range(n)]
    return {"chunks": chunks, "frames_total": total, "start": f0 / fps, "duration": total / fps}


def chunk_path(store: JobStore, rid: str, index: int) -> str:
    return str(store.render_dir(rid, create=True) / f"chunk_{index:03d}.mp4")


def render_chunk(engine: FaceEngine, store: JobStore, rid: str, chunk: dict, req: RenderRequest, progress) -> str:
    jid = rid.split("-")[0]
    tier = get_tier(req.tier)
    src, meta = str(store.video_path(jid)), store.read_json(jid, "meta.json")
    roles = load_roles(engine, store, jid, req.cast)
    out_path = chunk_path(store, rid, chunk["index"])
    key = f"render:{rid}:chunk:{chunk['index']}"
    count = chunk["count"]

    swapper = FrameSwapper(engine, tier, roles, meta["fps"], req.vr)
    reader = FrameReader(src, meta, chunk["start_frame"], count)
    writer = FrameWriter(out_path, meta, tier.crf, tier.preset)
    workers = 3 if engine.gpu else 1
    started = time.time()
    last_report, last_live, done, hits = 0.0, 0.0, 0, 0
    role_hits = [0] * len(roles)  # frames each of your faces was placed in

    def report(**extra):
        progress.put(key, {"done": done, "total": count, "swapped": hits, "roles": role_hits,
                           "started": started, "updated": time.time(), **extra})

    report()
    try:
        with ThreadPoolExecutor(workers) as pool:
            frames = iter(reader)
            pending: deque = deque()

            def feed():
                f = next(frames, None)
                if f is not None:
                    pending.append((f, pool.submit(swapper.analyze, f)))

            for _ in range(workers * 2):
                feed()
            while pending:
                frame, fut = pending.popleft()
                feed()
                out, placed = swapper.apply(frame, fut.result(), done)
                writer.write(out)
                done += 1
                hits += 1 if placed else 0
                for r in set(placed):
                    role_hits[r] += 1
                now = time.time()
                if now - last_report > 0.75:
                    last_report = now
                    report()
                if now - last_live > 2.0:
                    last_live = now
                    progress.put(f"render:{rid}:live", jpeg(out, max_w=640, quality=72))
    finally:
        reader.close()
        writer.close()
    report(finished=True)
    return out_path


def finalize_render(store: JobStore, rid: str, plan: dict) -> str:
    jid = rid.split("-")[0]
    src, meta = str(store.video_path(jid)), store.read_json(jid, "meta.json")
    rdir = store.render_dir(rid, create=True)
    chunks = [str(rdir / f"chunk_{c['index']:03d}.mp4") for c in plan["chunks"]]
    missing = [c for c in chunks if not os.path.exists(c)]
    if missing:
        raise RuntimeError(f"{len(missing)} render segment(s) went missing. Try rendering again.")
    final = str(rdir / "final.mp4")
    concat_and_mux(chunks, src, meta, plan["start"], plan["duration"], final)
    for c in chunks:
        os.remove(c)
    return final


def start_state(rid: str, req: RenderRequest, plan: dict) -> dict:
    return {"state": "queued", "render_id": rid, "request": asdict(req), "created": time.time(),
            "frames_total": plan["frames_total"], "duration": plan["duration"],
            "chunk_sizes": [c["count"] for c in plan["chunks"]]}


def validate_request(store: JobStore, jid: str, req: RenderRequest) -> None:
    for c in req.cast:
        if not face_embeddings(store, jid, c["face"]):
            raise ValueError(f"Add a clear photo for Face {c['face'].upper()} first.")
        load_target(store, jid, c["person"])


def run_render(store: JobStore, progress, rid: str, req: RenderRequest, plan: dict, run_chunks,
               sync=lambda: None, persist=lambda: None) -> None:
    """Drive one render: fan chunks out via `run_chunks`, then stitch + mux."""
    key = f"render:{rid}"
    state = progress.get(key) or start_state(rid, req, plan)
    state["state"] = "rendering"
    progress.put(key, state)
    try:
        validate_request(store, rid.split("-")[0], req)
        run_chunks(plan["chunks"])
        state["state"] = "finalizing"
        progress.put(key, state)
        sync()
        final = finalize_render(store, rid, plan)
        persist()
        state.update(state="done", finished=time.time(), size=os.path.getsize(final))
    except Exception as e:  # noqa: BLE001 - shown to the user
        state.update(state="error", error=str(e) or e.__class__.__name__)
    progress.put(key, state)


def render_status(progress, rid: str) -> dict | None:
    st = progress.get(f"render:{rid}")
    if st is None:
        return None
    now = time.time()
    lanes = []
    for i, size in enumerate(st["chunk_sizes"]):
        c = progress.get(f"render:{rid}:chunk:{i}") or {}
        lanes.append({"done": min(c.get("done", 0), size), "total": size, "active": "started" in c,
                      "started": c.get("started"), "swapped": c.get("swapped", 0), "roles": c.get("roles", [])})
    done = sum(l["done"] for l in lanes)
    total = max(1, st["frames_total"])
    starts = [l["started"] for l in lanes if l["started"]]
    eta = None
    if st["state"] == "rendering" and done > 0 and starts:
        rate = done / max(now - min(starts), 1e-3)
        eta = (total - done) / rate
    swapped = sum(l["swapped"] for l in lanes)
    cast = st["request"]["cast"]
    role_frames = [sum(l["roles"][r] for l in lanes if len(l["roles"]) > r) for r in range(len(cast))]
    end = st.get("finished", now)
    return {
        "state": st["state"], "error": st.get("error"), "request": st["request"],
        "frames_done": done, "frames_total": st["frames_total"], "progress": done / total,
        "eta": eta, "elapsed": end - st["created"], "duration": st["duration"],
        "lanes": [{"done": l["done"], "total": l["total"], "active": l["active"]} for l in lanes],
        "face_coverage": (swapped / done) if done else None, "size": st.get("size"),
        "role_coverage": [{"face": c["face"], "coverage": (f / done) if done else None}
                          for c, f in zip(cast, role_frames)],
    }
