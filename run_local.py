"""Run Lead Role on your own machine.

    pip install -r requirements.txt
    python run_local.py            # then open http://127.0.0.1:8000

Uses an NVIDIA GPU automatically if onnxruntime-gpu is installed, otherwise the CPU
(CPU works but is slow - fine for previews, expect minutes per second of video).
"""
import argparse
import os

import uvicorn

from faceswap.jobs import JobStore
from faceswap.local import LocalBackend
from faceswap.models import ensure_models
from faceswap.server import create_app


def main():
    ap = argparse.ArgumentParser(description="Lead Role face-swap studio (local)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", default=os.environ.get("FACESWAP_MODEL_DIR", "models"))
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    ensure_models(args.models)
    store = JobStore(os.path.join(args.data, "jobs"))
    app = create_app(LocalBackend(store, args.models), store)
    print(f"\n  Lead Role is running: http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
