#!/usr/bin/env sh
set -eu

SERVICE_NAME=lerobot-dataconvert.service
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
UNIT_DIR=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
UNIT_PATH=$UNIT_DIR/$SERVICE_NAME
TEMP_UNIT=$(mktemp "${TMPDIR:-/tmp}/lerobot-dataconvert.XXXXXX")
trap 'rm -f "$TEMP_UNIT"' EXIT HUP INT TERM

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "A running systemd user session is required." >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
cat >"$TEMP_UNIT" <<EOF
[Unit]
Description=LeRobot Data Convert workbench

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/start.sh
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
install -m 0644 "$TEMP_UNIT" "$UNIT_PATH"

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user stop "$SERVICE_NAME"

# Gracefully hand over a server that was started manually from this checkout.
if pgrep -f -- "$PROJECT_DIR/run.py" >/dev/null 2>&1; then
  pkill -TERM -f -- "$PROJECT_DIR/run.py"
  attempts=0
  while pgrep -f -- "$PROJECT_DIR/run.py" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge 30 ]; then
      echo "The existing backend did not stop within 30 seconds." >&2
      exit 1
    fi
    sleep 1
  done
fi

systemctl --user start "$SERVICE_NAME"
systemctl --user --no-pager --full status "$SERVICE_NAME"
