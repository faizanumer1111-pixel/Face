#!/usr/bin/env bash
# Start FaceSwap Studio and open it in your browser (http://127.0.0.1:7860).
cd "$(dirname "$0")"
for d in /opt/homebrew/bin /usr/local/bin; do
  case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$d:$PATH" ;; esac
done
export PATH
if [ ! -x .venv/bin/python ]; then
  echo "FaceSwap Studio isn't set up yet. Run:  ./scripts/setup.sh"
  exit 1
fi
exec .venv/bin/python -m app.server "$@"
