#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_PYTHON=$PROJECT_DIR/.venv/bin/python
if [ ! -x "$DEFAULT_PYTHON" ]; then
  DEFAULT_PYTHON=/home/amin/miniconda3/envs/lerobot21/bin/python
fi
PYTHON=${LEROBOT_DATACONVERT_PYTHON:-$DEFAULT_PYTHON}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python3
fi

exec "$PYTHON" "$PROJECT_DIR/run.py" "$@"
