#!/usr/bin/env sh
set -eu

SERVICE_NAME=lerobot-dataconvert.service
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BACKEND_URL=${LEROBOT_DATACONVERT_URL:-http://127.0.0.1:8765}
DEFAULT_PYTHON=$PROJECT_DIR/.venv/bin/python
if [ ! -x "$DEFAULT_PYTHON" ]; then
  DEFAULT_PYTHON=/home/amin/miniconda3/envs/lerobot21/bin/python
fi
PYTHON=${LEROBOT_DATACONVERT_PYTHON:-$DEFAULT_PYTHON}
if [ ! -x "$PYTHON" ]; then
  PYTHON=$(command -v python3 || true)
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo "Python 3 was not found. Follow INSTALL.md before updating." >&2
  exit 1
fi
if [ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]; then
  echo "Local changes detected. Ask for technical help before applying the update." >&2
  exit 1
fi
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "A running systemd user session is required." >&2
  exit 1
fi
if ! systemctl --user cat "$SERVICE_NAME" >/dev/null 2>&1; then
  echo "Service not installed. Run ./install-systemd-service.sh first." >&2
  exit 1
fi

ensure_idle() {
  if ! systemctl --user is-active --quiet "$SERVICE_NAME"; then
    return
  fi
  "$PYTHON" - "$BACKEND_URL/api/jobs" <<'PY'
import json
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=3) as response:
        jobs = json.load(response).get("jobs")
except (OSError, ValueError) as exc:
    print(f"Could not read active jobs: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(jobs, list):
    print("Could not read active jobs: invalid API response", file=sys.stderr)
    raise SystemExit(1)
active_states = {"queued", "running", "merging", "stopping"}
active = [str(job.get("id", "unknown")) for job in jobs if job.get("state") in active_states]
if active:
    print(f"Active conversion jobs detected: {', '.join(active)}. Wait or stop them first.", file=sys.stderr)
    raise SystemExit(1)
PY
}

health_ok() {
  "$PYTHON" - "$BACKEND_URL/api/health" <<'PY'
import json
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        healthy = json.load(response).get("ok") is True
except (OSError, ValueError):
    healthy = False
raise SystemExit(0 if healthy else 1)
PY
}

ensure_idle
echo "[1/3] Installing declared Python dependencies..."
"$PYTHON" -m pip install -e "$PROJECT_DIR"

ensure_idle
echo "[2/3] Restarting $SERVICE_NAME..."
systemctl --user restart "$SERVICE_NAME"

echo "[3/3] Waiting for backend health..."
attempts=0
until health_ok; do
  attempts=$((attempts + 1))
  if [ "$attempts" -ge 30 ]; then
    echo "Backend did not become healthy. Check: journalctl --user -u lerobot-dataconvert -n 100" >&2
    exit 1
  fi
  sleep 1
done

echo "Update applied successfully: $BACKEND_URL"
