#!/bin/bash
# Lead Role: one-command install + launch on a Mac (Apple silicon or Intel).
#
#   curl -fsSL https://raw.githubusercontent.com/faizanumer1111-pixel/Face/claude/practical-einstein-7yvfy4/mac_install.sh | bash
#
# Installs into ~/LeadRole, downloads the face models (~1.2 GB) on first run, then opens
# http://127.0.0.1:8000 in your browser. Run it again any time to update and relaunch.
set -euo pipefail

BRANCH="${LEADROLE_BRANCH:-claude/practical-einstein-7yvfy4}"
REPO="https://github.com/faizanumer1111-pixel/Face.git"
DIR="${LEADROLE_DIR:-$HOME/LeadRole}"
PORT="${LEADROLE_PORT:-8000}"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mProblem: %s\033[0m\n' "$*" >&2; exit 1; }

# 1. A Python 3.10+ (macOS's built-in python3 is often 3.9, which is too old)
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing Python 3.12 with Homebrew"
    brew install python@3.12
    PY="$(brew --prefix python@3.12)/bin/python3.12"
  else
    die "Python 3.10 or newer is needed. Install the latest macOS version from https://www.python.org/downloads/ and run this command again."
  fi
fi
say "Using $("$PY" --version) at $PY"

# 2. Command line tools (compiler for one small extension)
if ! xcode-select -p >/dev/null 2>&1; then
  say "Installing Apple's command line tools (a dialog will open; click Install, then run this command again)"
  xcode-select --install || true
  exit 0
fi

# 3. Get the code
mkdir -p "$DIR"
if [ -d "$DIR/app/.git" ]; then
  say "Updating Lead Role"
  git -C "$DIR/app" fetch -q origin "$BRANCH" && git -C "$DIR/app" checkout -q "$BRANCH" && git -C "$DIR/app" reset -q --hard "origin/$BRANCH"
else
  say "Downloading Lead Role"
  git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$DIR/app"
fi

# 4. Python environment + packages
if [ ! -x "$DIR/env/bin/python" ]; then
  say "Creating Python environment"
  "$PY" -m venv "$DIR/env"
fi
VPY="$DIR/env/bin/python"
say "Installing packages (first time takes a few minutes)"
"$VPY" -m pip install -q --upgrade pip wheel setuptools
"$VPY" -m pip install -q "numpy<2.3" cython
"$VPY" -m pip install -q --no-build-isolation insightface==0.7.3
"$VPY" -m pip install -q opencv-python-headless onnxruntime fastapi "uvicorn[standard]" python-multipart static-ffmpeg
"$VPY" -c "import static_ffmpeg; static_ffmpeg.add_paths()" >/dev/null 2>&1 || true

# 5. Launch under a watchdog: restarts the app if it ever stops, keeps the Mac awake while running,
#    and unfinished renders resume on their own.
say "Starting Lead Role (face models download on the first run, ~1.2 GB)"
pkill -f "start_local.sh" >/dev/null 2>&1 || true
pkill -f "run_local.py" >/dev/null 2>&1 || true
cd "$DIR/app"
chmod +x start_local.sh
LEADROLE_PYTHON="$VPY" LEADROLE_LOG="$DIR/server.log" nohup ./start_local.sh --port "$PORT" --models "$DIR/models" --data "$DIR/data" >/dev/null 2>&1 &
for _ in $(seq 1 600); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then break; fi
  sleep 1
done
if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
  say "Lead Role is running at http://127.0.0.1:$PORT"
  open "http://127.0.0.1:$PORT"
  echo "To stop it later:  pkill -f start_local.sh; pkill -f run_local.py"
  echo "Log file:          $DIR/server.log"
else
  die "The app didn't start. Send the last lines of $DIR/server.log to Claude:  tail -30 $DIR/server.log"
fi
