"""Time each model on every available provider and keep the fastest one that's correct.

On a Mac this compares CoreML (Apple GPU / Neural Engine) with the CPU. A provider
only wins if its output matches the CPU's, so a provider that runs but returns
wrong numbers is never picked. The choice is saved to data/settings.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .faces import DET_MODEL, REC_MODEL, SWAP_MODEL, available_provider_choices, make_session, settings_path

MODELS = {"det": DET_MODEL, "rec": REC_MODEL, "swap": SWAP_MODEL}


def _inputs(key: str, sess) -> dict:
    rng = np.random.default_rng(0)
    ins = sess.get_inputs()
    if key == "det":
        return {ins[0].name: rng.uniform(-1, 1, (1, 3, 640, 640)).astype(np.float32)}
    if key == "rec":
        return {ins[0].name: rng.uniform(-1, 1, (1, 3, 112, 112)).astype(np.float32)}
    lat = rng.normal(size=(1, 512)).astype(np.float32)
    lat /= np.linalg.norm(lat)
    return {ins[0].name: rng.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32), ins[1].name: lat}


def _det_small(sess) -> dict:
    rng = np.random.default_rng(1)
    return {sess.get_inputs()[0].name: rng.uniform(-1, 1, (1, 3, 320, 320)).astype(np.float32)}


def _matches(key: str, out: list, ref: list) -> tuple[bool, str]:
    if key == "rec":
        a, b = out[0].ravel(), ref[0].ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return cos > 0.995, f"cosine {cos:.4f}"
    if key == "swap":
        err = float(np.abs(out[0] - ref[0]).mean())
        return err < 0.01, f"mean error {err:.4f}"
    worst = max(float(np.abs(o - r).max()) for o, r in zip(out, ref))
    return worst < 0.05, f"max error {worst:.4f}"


def run_benchmark(models_dir: Path, data_dir: Path, runs: int = 20) -> dict:
    choices = available_provider_choices()
    cache = models_dir / "coreml-cache"
    report: dict = {"available": choices, "models": {}}
    picked: dict[str, str] = {}
    for key, fname in MODELS.items():
        path = models_dir / fname
        results = {}
        ref_out = None
        ref_small = None
        feeds = None
        for choice in ["cpu"] + [c for c in choices if c != "cpu"]:
            entry: dict = {}
            try:
                t0 = time.perf_counter()
                sess, active = make_session(path, choice, cache)
                entry["load_seconds"] = round(time.perf_counter() - t0, 2)
                if active != choice:
                    entry["error"] = f"{choice} couldn't load this model"
                    results[choice] = entry
                    continue
                feeds = feeds or _inputs(key, sess)
                out = sess.run(None, feeds)
                for _ in range(2):
                    sess.run(None, feeds)
                t0 = time.perf_counter()
                for _ in range(runs):
                    sess.run(None, feeds)
                ms = (time.perf_counter() - t0) / runs * 1000
                entry["ms_per_run"] = round(ms, 2)
                # The detector also runs at 320 px ("Fast" detection): check that size too.
                small = sess.run(None, _det_small(sess)) if key == "det" else None
                if choice == "cpu":
                    ref_out, ref_small = out, small
                    entry["correct"] = True
                else:
                    ok, detail = _matches(key, out, ref_out)
                    if ok and small is not None:
                        ok, detail = _matches(key, small, ref_small)
                    entry["correct"] = ok
                    entry["check"] = detail
            except Exception as exc:  # report and move on
                entry["error"] = str(exc).splitlines()[0][:200]
            results[choice] = entry
        valid = {c: e["ms_per_run"] for c, e in results.items() if e.get("correct") and "ms_per_run" in e}
        picked[key] = min(valid, key=valid.get) if valid else "cpu"
        report["models"][key] = {"file": fname, "results": results, "picked": picked[key]}

    per_face = sum(report["models"][k]["results"][picked[k]]["ms_per_run"] for k in ("det", "swap"))
    report["estimated_fps_one_face"] = round(1000 / per_face, 1) if per_face else None
    report["providers"] = picked
    data_dir.mkdir(parents=True, exist_ok=True)
    p = settings_path(data_dir)
    current = json.loads(p.read_text()) if p.exists() else {}
    current["providers"] = picked
    current["benchmark"] = report
    p.write_text(json.dumps(current, indent=2))
    return report
