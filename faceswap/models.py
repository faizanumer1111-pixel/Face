"""Download the face models.  Usage:  python -m faceswap.models ./models"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import MODEL_URLS

REQUIRED = ["inswapper_128.onnx", "gfpgan_1.4.onnx", "bisenet_resnet_34.onnx",
            "buffalo_l/det_10g.onnx", "buffalo_l/w600k_r50.onnx"]


def _fetch(url: str) -> bytes:
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def ensure_models(dest: str | Path) -> Path:
    dest = Path(dest)
    (dest / "buffalo_l").mkdir(parents=True, exist_ok=True)
    for name in ("inswapper_128.onnx", "gfpgan_1.4.onnx", "bisenet_resnet_34.onnx"):
        if not (dest / name).exists():
            tmp = dest / (name + ".part")
            tmp.write_bytes(_fetch(MODEL_URLS[name]))
            tmp.replace(dest / name)
    if not all((dest / "buffalo_l" / n).exists() for n in ("det_10g.onnx", "w600k_r50.onnx")):
        with zipfile.ZipFile(io.BytesIO(_fetch(MODEL_URLS["buffalo_l.zip"]))) as z:
            for member in z.namelist():
                base = member.rsplit("/", 1)[-1]
                if base in ("det_10g.onnx", "w600k_r50.onnx"):
                    (dest / "buffalo_l" / base).write_bytes(z.read(member))
    missing = [r for r in REQUIRED if not (dest / r).exists()]
    if missing:
        raise RuntimeError(f"Model download incomplete, missing: {missing}")
    return dest


if __name__ == "__main__":
    out = ensure_models(sys.argv[1] if len(sys.argv) > 1 else "models")
    print(f"Models ready in {out}")
