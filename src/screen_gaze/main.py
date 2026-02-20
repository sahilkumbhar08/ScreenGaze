#!/usr/bin/env python3
"""ScreenGaze entry point: launch GUI or CLI."""
from __future__ import annotations

import importlib.resources
import sys
from pathlib import Path
import argparse
from screen_gaze.gui.main_window import MainWindow
from screen_gaze.core.face_tracker import list_monitors, run

def main() -> None:
    parser = argparse.ArgumentParser(description="ScreenGaze - Gaze-based screen tracking")
    parser.add_argument("--no-gui", action="store_true", dest="no_gui", help="Run CLI with OpenCV windows instead of GUI")
    parser.add_argument("--config", type=Path, default=importlib.resources.files("screen_gaze.conf") / "config.json")
    parser.add_argument("--list-monitors", action="store_true", help="Print monitor layout and exit")
    parser.add_argument("--no-preview", action="store_true", help="(CLI only) Run without camera preview")
    parser.add_argument("--calibrate", action="store_true", help="(CLI only) Run calibration wizard then start")
    args = parser.parse_args()

    if args.list_monitors:
        list_monitors()
        return

    if args.no_gui:
        run(args.config, no_preview=args.no_preview, force_calibrate=args.calibrate)
        return

    app = MainWindow(args.config)
    app.run()


if __name__ == "__main__":
    main()
