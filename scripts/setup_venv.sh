#!/usr/bin/env bash
set -euo pipefail

# Setup a Python 3.11 virtual environment and install dependencies.
# If python3.11 is not available, the script will try `python3` but will warn.

PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "ERROR: python3 not found. Install Python 3.11 or set PYTHON env var." >&2
    exit 1
  fi
fi

echo "Using interpreter: $($PYTHON --version 2>&1)"

VEV_DIR=".venv"
if [ ! -d "$VEV_DIR" ]; then
  $PYTHON -m venv "$VEV_DIR"
fi

# Activate and install
# shellcheck disable=SC1091
source "$VEV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Virtual environment ready. Activate with: source $VEV_DIR/bin/activate"
