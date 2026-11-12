#!/usr/bin/env bash
# One-time setup for FaceSwap Studio on a Mac (Apple Silicon, macOS 14 or newer).
#   1. installs ffmpeg and Python 3.12 with Homebrew (if missing)
#   2. creates the Python environment in .venv
#   3. downloads the face models (~1.3 GB) and checks their SHA-256
#   4. measures whether the Mac's GPU (CoreML) or CPU is faster, and saves the choice
# Safe to run again: it skips whatever is already done.
set -euo pipefail
cd "$(dirname "$0")/.."

INSWAPPER_URL="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
INSWAPPER_SHA="a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7"
BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_SHA="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
DET_SHA="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
REC_SHA="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
# Sharper faces (GFPGAN 1.4) and hands-in-front (XSeg occlusion), ONNX versions from FaceFusion's assets
GFPGAN_URL="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx"
GFPGAN_SHA="accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"
XSEG_URL="https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx"
XSEG_SHA="c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3"

say() { printf "\n\033[1m%s\033[0m\n" "$*"; }
fail() { printf "\n\033[31mSetup stopped: %s\033[0m\n" "$*" >&2; exit 1; }
sha() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }

for d in /opt/homebrew/bin /usr/local/bin; do
  case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$d:$PATH" ;; esac
done
export PATH

OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  [ "$(uname -m)" = "arm64" ] || fail "This needs an Apple Silicon Mac (M1 or newer)."
  MACOS="$(sw_vers -productVersion)"
  [ "${MACOS%%.*}" -ge 14 ] || fail "This needs macOS 14 (Sonoma) or newer. This Mac has macOS $MACOS."
  command -v brew >/dev/null 2>&1 || fail "Homebrew isn't installed. Install it from https://brew.sh (copy the one-line command on that page into Terminal), then run this script again."
  say "1/4  Installing ffmpeg and Python 3.12 with Homebrew (skips what's already there)…"
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
  brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
  PY="$(brew --prefix python@3.12)/bin/python3.12"
else
  say "1/4  Checking ffmpeg and Python 3.12…"
  command -v ffmpeg >/dev/null 2>&1 || fail "Install ffmpeg first."
  PY="$(command -v python3.12 || true)"
  [ -n "$PY" ] || fail "Install Python 3.12 first."
fi
echo "  ffmpeg: $(ffmpeg -hide_banner -version | head -1 | cut -d' ' -f3)  python: $("$PY" --version)"

say "2/4  Setting up the Python environment (.venv)…"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "  done"

say "3/4  Downloading the face models (about 1.3 GB, only the first time)…"
mkdir -p models
fetch() {
  local url="$1" dest="$2" want="$3" got
  if [ -f "$dest" ] && [ "$(sha "$dest")" = "$want" ]; then
    echo "  $(basename "$dest"): already downloaded"
    return
  fi
  echo "  $(basename "$dest"):"
  curl -fL --retry 4 --retry-delay 3 --progress-bar -o "$dest.part" "$url" \
    || fail "Couldn't download $(basename "$dest"). Check your internet connection and run this again."
  got="$(sha "$dest.part")"
  if [ "$got" != "$want" ]; then
    rm -f "$dest.part"
    fail "$(basename "$dest") didn't match its expected checksum. Run this again; if it keeps failing, the download source has changed."
  fi
  mv "$dest.part" "$dest"
}
fetch "$INSWAPPER_URL" models/inswapper_128.onnx "$INSWAPPER_SHA"
if [ ! -f models/det_10g.onnx ] || [ "$(sha models/det_10g.onnx)" != "$DET_SHA" ] \
   || [ ! -f models/w600k_r50.onnx ] || [ "$(sha models/w600k_r50.onnx)" != "$REC_SHA" ]; then
  fetch "$BUFFALO_URL" models/buffalo_l.zip "$BUFFALO_SHA"
  unzip -o -j -q models/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d models
  rm -f models/buffalo_l.zip
  [ "$(sha models/det_10g.onnx)" = "$DET_SHA" ] && [ "$(sha models/w600k_r50.onnx)" = "$REC_SHA" ] \
    || fail "The face detector files didn't match their checksums."
else
  echo "  det_10g.onnx, w600k_r50.onnx: already downloaded"
fi
fetch "$GFPGAN_URL" models/gfpgan_1.4.onnx "$GFPGAN_SHA"
fetch "$XSEG_URL" models/xseg_1.onnx "$XSEG_SHA"

say "4/4  Measuring speed: Mac GPU (CoreML) vs CPU. The first CoreML run compiles the models and can take a minute…"
if ! .venv/bin/python -m app.cli benchmark --runs 10; then
  rm -f data/settings.json
  echo "  Couldn't measure the GPU, so FaceSwap Studio will use the CPU (it works, just slower)."
fi

say "All set. Start FaceSwap Studio with:  ./run.sh   (or double-click \"FaceSwap Studio.command\")"
