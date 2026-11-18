#!/bin/bash
# Keeps Lead Role running: restarts it if it ever stops, and (on a Mac) stops the
# computer from sleeping while it runs. Unfinished renders resume automatically.
#
#   ./start_local.sh [--port 8000] [--models models] [--data data]
#
# Stop it with:  pkill -f start_local.sh; pkill -f run_local.py
set -u
cd "$(dirname "$0")"
PY="${LEADROLE_PYTHON:-python3}"
[ -x "env/bin/python" ] && PY="env/bin/python"
[ -x "../env/bin/python" ] && PY="../env/bin/python"
LOG="${LEADROLE_LOG:-leadrole.log}"
KEEP_AWAKE=""
command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE="caffeinate -i"

echo "Lead Role watchdog started $(date)" >> "$LOG"
while true; do
  $KEEP_AWAKE "$PY" run_local.py "$@" >> "$LOG" 2>&1
  code=$?
  echo "Lead Role stopped (exit $code) at $(date); restarting in 3 s" >> "$LOG"
  sleep 3
done
