#!/usr/bin/env bash
# ScreenGaze launcher: run from anywhere.
# Usage: ./run.sh [options]
#
# Actions:
#   - cd to script directory (so config.json, gui/, etc. are found)
#   - activate .venv if present
#   - run main.py (GUI by default)
#
# Requirements: Python 3, tkinter, xdotool, 2+ monitors.
# Install deps: pip install -r requirements.txt
#
# Options (e.g. ./run.sh --no-gui):
#   --no-gui          CLI with OpenCV windows
#   --list-monitors   Print monitor layout and exit
#   --no-preview      (with --no-gui) No camera window
#   --calibrate       (with --no-gui) Run calibration then start

set -e
cd "$(dirname "$0")"

if [[ -d .venv ]]; then
  . .venv/bin/activate
fi

exec python main.py "$@"
