#!/usr/bin/env python3
"""Multi-screen face tracker: look at a screen → cursor moves there."""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from screeninfo import get_monitors

CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_CAPTURE_SEC = 1.5
CALIBRATION_MIN_SAMPLES = 20

# BGR colors matching gui/theme.py (dark mode) for calibration and panels
_CAL_BG_DARK = (36, 29, 26)      # #1a1d24
_CAL_BG_CARD = (51, 42, 37)      # #252a33
_CAL_BG_INPUT = (61, 51, 45)     # #2d333d
_CAL_BORDER = (83, 69, 61)       # #3d4553
_CAL_ACCENT = (196, 205, 78)     # #4ecdc4
_CAL_TEXT = (237, 234, 232)      # #e8eaed
_CAL_TEXT_DIM = (166, 160, 154) # #9aa0a6
_CAL_KEY_BG = (61, 51, 45)       # same as input
_CAL_KEY_FG = (200, 230, 220)   # light accent for key letters


def _draw_text_panel(frame, lines: list[str], y_start: int = 60, line_height: int = 60, font_scale: float = 1.1, thickness: int = 2, title: str | None = None):
    """Draw readable text on a dark panel with styling matching the main GUI theme."""
    import cv2
    h, w = frame.shape[:2]
    n = len(lines)
    pad_x = 40
    pad_y = 35
    
    # Calculate panel dimensions
    panel_w = min(w - 60, 700)
    x1 = (w - panel_w) // 2
    x2 = x1 + panel_w
    
    # Calculate total height needed
    total_height = pad_y * 2
    if title:
        total_height += line_height + 20
    total_height += n * line_height
    
    # Ensure panel fits within frame
    y1 = min(y_start, h - total_height - 20)
    y1 = max(20, y1)
    y2 = y1 + total_height
    
    # Panel background and border (match GUI card)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _CAL_BG_CARD, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _CAL_ACCENT, 2)
    
    # Draw title if provided
    current_y = y1 + pad_y
    if title:
        title_scale = font_scale * 1.3
        title_thickness = thickness + 1
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, title_thickness)
        title_x = x1 + (panel_w - title_w) // 2
        title_y = current_y + title_h
        
        # Title (theme text color, no special chars)
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, title_scale, (0, 0, 0), title_thickness + 2)
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, title_scale, _CAL_TEXT, title_thickness)
        
        # Underline (accent)
        line_y = title_y + 12
        cv2.line(frame, (x1 + 60, line_y), (x2 - 60, line_y), _CAL_ACCENT, 2)
        current_y = line_y + 25
    
    # Draw lines with better formatting
    for i, line in enumerate(lines):
        if not line or not line.strip():
            continue
        
        y = current_y + i * line_height
        
        # Check if line has a shortcut key format like "C   Description" or "SPACE   Description"
        stripped = line.strip()
        if stripped.startswith("SPACE  ") or stripped.startswith("SPACE "):
            key = "SPACE"
            desc = stripped[5:].lstrip()
            key_x = x1 + pad_x
            key_w = 72
            key_h = line_height - 10
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_KEY_BG, -1)
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_ACCENT, 1)
            (k_w, k_h), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, thickness)
            kx = key_x + (key_w - k_w) // 2
            ky = y - (key_h // 2) + 3
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (0, 0, 0), thickness + 2)
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, _CAL_KEY_FG, thickness)
            if desc:
                desc_x = key_x + key_w + 20
                cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _CAL_TEXT, thickness)
        elif len(stripped) >= 2 and stripped[0] in 'CSD' and stripped[1:3] == '  ':
            # Split into key and description
            key = stripped[0]
            desc = stripped[2:].lstrip()
            
            # Draw key in bright yellow/orange with background
            key_x = x1 + pad_x
            key_w = 35
            key_h = line_height - 10
            
            # Key background (match GUI button style)
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_KEY_BG, -1)
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_ACCENT, 1)
            
            # Draw key letter centered (ASCII only)
            (k_w, k_h), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, thickness + 1)
            kx = key_x + (key_w - k_w) // 2
            ky = y - (key_h // 2) + 3
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (0, 0, 0), thickness + 2)
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, _CAL_KEY_FG, thickness + 1)
            
            # Description (theme text)
            desc_x = key_x + key_w + 20
            cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _CAL_TEXT, thickness)
        elif stripped.startswith('ESC'):
            # ESC key (same style)
            key = 'ESC'
            desc = stripped[3:].lstrip() if len(stripped) > 3 else ''
            
            key_x = x1 + pad_x
            key_w = 60
            key_h = line_height - 10
            
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_KEY_BG, -1)
            cv2.rectangle(frame, (key_x, y - key_h + 5), (key_x + key_w, y + 5), _CAL_ACCENT, 1)
            
            (k_w, k_h), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, thickness)
            kx = key_x + (key_w - k_w) // 2
            ky = y - (key_h // 2) + 3
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 0, 0), thickness + 2)
            cv2.putText(frame, key, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, _CAL_KEY_FG, thickness)
            
            if desc:
                desc_x = key_x + key_w + 20
                cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, desc, (desc_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _CAL_TEXT, thickness)
        else:
            # Regular line (ASCII only)
            cv2.putText(frame, line, (x1 + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, line, (x1 + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _CAL_TEXT, thickness)


def move_cursor(x: int, y: int) -> bool:
    xdotool = shutil.which("xdotool")
    if not xdotool:
        return False
    try:
        subprocess.run(
            [xdotool, "mousemove", "--sync", str(x), str(y)],
            check=True,
            capture_output=True,
            timeout=2,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_config(path: Path, config: dict) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def get_ordered_monitors(order: str) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) for each screen as left, middle, right. order 'auto' = by position."""
    monitors = list(get_monitors())
    if len(monitors) < 2:
        return [(m.x, m.y, m.width, m.height) for m in monitors]

    if order == "auto":
        by_x = sorted(monitors, key=lambda m: m.x)
        # Middle screen is the one "down" (largest y) if not in a single row
        ys = [m.y for m in by_x]
        if max(ys) - min(ys) > 50:  # different rows
            # Bottom row = middle
            middle = max(monitors, key=lambda m: m.y)
            left_right = [m for m in by_x if m is not middle]
            # Left = smaller x, right = larger x
            left_right.sort(key=lambda m: m.x)
            ordered = [left_right[0], middle, left_right[1]] if len(left_right) == 2 else by_x
        else:
            ordered = by_x  # left, middle, right by x
        return [(m.x, m.y, m.width, m.height) for m in ordered]

    by_x = sorted(monitors, key=lambda m: m.x)
    indices = [int(i) for i in order.split(",")]
    ordered = [by_x[i] for i in indices if 0 <= i < len(by_x)]
    return [(m.x, m.y, m.width, m.height) for m in ordered]


def center_of_region(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return (x + w // 2, y + h // 2)


NOSE_TIP_IDX = 4
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _model_path() -> Path:
    project_dir = Path(__file__).resolve().parent
    path = project_dir / "face_landmarker.task"
    if path.exists():
        return path
    import urllib.request
    print("Downloading face_landmarker model (one-time)...", flush=True)
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    print("Done.", flush=True)
    return path


def init_face_mesh():
    import mediapipe as mp
    model_path = _model_path()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def project_dir() -> Path:
    """Always the directory containing this script (for calibration.json)."""
    return Path(__file__).resolve().parent


def calibration_path() -> Path:
    return project_dir() / CALIBRATION_FILENAME


def load_calibration(num_screens: int) -> list[float] | None:
    path = calibration_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        refs = data.get("refs")
        if isinstance(refs, list) and len(refs) == num_screens and all(isinstance(r, (int, float)) for r in refs):
            return [float(r) for r in refs]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_calibration(refs: list[float], num_screens: int) -> None:
    path = calibration_path()
    with open(path, "w") as f:
        json.dump({"refs": refs, "num_screens": num_screens}, f, indent=2)


def get_head_turn_norm_x(frame, face_landmarker, sensitivity: float = 1.0, frame_timestamp_ms: int = 0) -> float | None:
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if not result.face_landmarks:
        return None
    lm_list = result.face_landmarks[0]
    nose = lm_list[NOSE_TIP_IDX]
    xs = [p.x for p in lm_list]
    left_x, right_x = min(xs), max(xs)
    face_center_x = (left_x + right_x) * 0.5
    face_half_width = max((right_x - left_x) * 0.5, 1e-6)
    yaw_proxy = (nose.x - face_center_x) / face_half_width
    yaw_proxy = max(-1.0, min(1.0, yaw_proxy * sensitivity))
    norm_x = 0.5 - 0.5 * yaw_proxy  # look left → left screen, look right → right
    return norm_x


def _center_window(window_name: str, width: int, height: int):
    """Center an OpenCV window on the primary monitor."""
    import cv2
    monitors = list(get_monitors())
    if monitors:
        primary = monitors[0]
        x = primary.x + (primary.width - width) // 2
        y = primary.y + (primary.height - height) // 2
    else:
        x, y = 100, 100
    cv2.moveWindow(window_name, x, y)


def show_calibration_choice(cap, num_screens: int) -> str:
    """Show pop-up: C=Calibrate, S=Use saved, D=Defaults. Returns 'calibrate'|'saved'|'defaults'."""
    import cv2
    window_name = "ScreenGaze - Setup"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, 900, 650)
    _center_window(window_name, 900, 650)
    has_saved = load_calibration(num_screens) is not None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), _CAL_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        lines = [
            "C   Calibrate now (map your face to each screen)",
            "S   Use saved calibration" if has_saved else "D   Defaults (no calibration)",
        ]
        if has_saved:
            lines.append("D   Defaults (no calibration)")
        lines.append("ESC   Exit")
        _draw_text_panel(frame, lines, y_start=150, line_height=65, font_scale=1.0, thickness=2, title="ScreenGaze Setup")
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(50)
        if key == -1:
            continue
        k = key & 0xFF
        if k == ord("c") or k == ord("C"):
            cv2.destroyWindow(window_name)
            return "calibrate"
        if has_saved and (k == ord("s") or k == ord("S")):
            cv2.destroyWindow(window_name)
            return "saved"
        if k == ord("d") or k == ord("D"):
            cv2.destroyWindow(window_name)
            return "defaults"
        if key == 27:
            cv2.destroyWindow(window_name)
            return "defaults"
    return "defaults"


def run_calibration(
    cap,
    face_landmarker,
    config: dict,
    cursor_positions: list[tuple[int, int]],
    scaled_pos,
    num_screens: int,
) -> list[float]:
    import cv2
    sensitivity = config.get("head_sensitivity", 1.15)
    fps = max(10, config.get("fps_limit", 24))
    frame_dt = 1.0 / fps
    labels = ["Left", "Right"] if num_screens == 2 else ["Left", "Middle", "Right"]
    refs: list[float] = []
    window_name = "ScreenGaze - Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, 900, 700)
    _center_window(window_name, 900, 700)
    capture_frames = max(CALIBRATION_MIN_SAMPLES, int(CALIBRATION_CAPTURE_SEC * fps))
    frame_ts_ms = 0
    ms_per_frame = max(1, int(1000 // fps))

    for step in range(num_screens):
        move_cursor(*scaled_pos(step))
        waiting = True
        while waiting:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_ts_ms += ms_per_frame
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _CAL_BG_DARK, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            _draw_text_panel(frame, [
                "Look at your " + labels[step] + " monitor",
                "SPACE   Capture your head position",
            ], y_start=180, line_height=70, font_scale=1.1, thickness=2, title="Step " + str(step + 1) + " of " + str(num_screens))
            nx = get_head_turn_norm_x(frame, face_landmarker, sensitivity, frame_ts_ms)
            if nx is None:
                _draw_text_panel(frame, ["[!] Position your face in the camera view"], y_start=h // 2 + 40, line_height=44, font_scale=0.85)
            # Status bar (match GUI theme)
            cv2.rectangle(frame, (0, h - 44), (w, h), _CAL_BG_CARD, -1)
            cv2.rectangle(frame, (0, h - 44), (w, h), _CAL_ACCENT, 1)
            cv2.putText(frame, "ESC   Cancel calibration", (30, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.68, _CAL_TEXT_DIM, 1)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30)
            if key != -1:
                k = key & 0xFF
                if k == ord(" "):
                    waiting = False
                elif k == 27:
                    cv2.destroyAllWindows()
                    raise SystemExit(0)

        samples: list[float] = []
        for c in range(capture_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_ts_ms += ms_per_frame
            nx = get_head_turn_norm_x(frame, face_landmarker, sensitivity, frame_ts_ms)
            if nx is not None:
                samples.append(nx)
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _CAL_BG_DARK, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
            progress = (c + 1) / capture_frames
            _draw_text_panel(frame, ["Hold still while capturing...", ""], y_start=80, line_height=46, font_scale=0.95, title="Capturing " + labels[step] + " Position")
            # Progress bar (match GUI slider/theme)
            bar_y = h // 2 + 40
            bar_height = 26
            bar_width = min(520, w - 160)
            bar_x = (w - bar_width) // 2
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), _CAL_BG_INPUT, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), _CAL_ACCENT, 1)
            fill_width = int(bar_width * progress)
            if fill_width > 4:
                cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_width - 2, bar_y + bar_height - 2), _CAL_ACCENT, -1)
            # Percentage text
            pct_text = str(int(progress * 100)) + "%"
            text_x = bar_x + (bar_width - len(pct_text) * 14) // 2
            cv2.putText(frame, pct_text, (text_x, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, _CAL_TEXT, 2)
            cv2.imshow(window_name, frame)
            cv2.waitKey(max(1, int(frame_dt * 1000)))

        if samples:
            refs.append(sum(samples) / len(samples))
        else:
            refs.append((step + 0.5) / num_screens)

    # Show "Calibration complete" (same theme as GUI)
    for _ in range(30):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), _CAL_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        _draw_text_panel(frame, [
            "All positions captured successfully",
            "You can start tracking from the main window.",
        ], y_start=200, line_height=70, font_scale=1.1, thickness=2, title="Calibration Complete")
        cv2.imshow(window_name, frame)
        if cv2.waitKey(50) == 27:
            break
    cv2.destroyAllWindows()
    save_calibration(refs, num_screens)
    return refs


def run_calibration_standalone(config_path: Path) -> None:
    """Run calibration wizard (camera + OpenCV windows). Use from GUI before starting tracking."""
    import cv2
    config = load_config(config_path)
    camera_index = config.get("camera_index", 0)
    frame_w = config.get("frame_width", 640)
    frame_h = config.get("frame_height", 480)
    screen_order = config.get("screen_order", "auto")
    cursor_scale = config.get("cursor_scale", 1.0)
    monitors = get_ordered_monitors(screen_order)
    if len(monitors) < 2:
        raise RuntimeError("Need at least 2 monitors. Found: " + str(len(monitors)))
    num_screens = len(monitors)
    cursor_positions = [center_of_region(*m) for m in monitors]

    def scaled_pos(i: int) -> tuple[int, int]:
        cx, cy = cursor_positions[i]
        return (int(cx * cursor_scale), int(cy * cursor_scale))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    face_mesh = init_face_mesh()
    try:
        choice = show_calibration_choice(cap, num_screens)
        if choice == "calibrate":
            run_calibration(cap, face_mesh, config, cursor_positions, scaled_pos, num_screens)
        elif choice == "saved":
            load_calibration(num_screens)  # just ensure it exists; refs saved to file
        # else: defaults, nothing to save
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_tracking_loop(
    config_path: Path,
    frame_queue: queue.Queue | None,
    stop_event: threading.Event,
    get_config: Callable[[], dict],
) -> None:
    """Run tracking in a loop; push frames to frame_queue and exit when stop_event is set.
    Use from GUI: run this in a daemon thread, pass get_config so GUI can update sensitivity/smoothing.
    """
    import cv2
    config = get_config()
    camera_index = config.get("camera_index", 0)
    frame_w = config.get("frame_width", 640)
    frame_h = config.get("frame_height", 480)
    fps_limit = config.get("fps_limit", 24)
    screen_order = config.get("screen_order", "auto")
    cursor_scale = config.get("cursor_scale", 1.0)
    monitors = get_ordered_monitors(screen_order)
    if len(monitors) < 2:
        return
    num_screens = len(monitors)
    cursor_positions = [center_of_region(*m) for m in monitors]

    def scaled_pos(i: int) -> tuple[int, int]:
        cx, cy = cursor_positions[i]
        return (int(cx * cursor_scale), int(cy * cursor_scale))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    face_mesh = init_face_mesh()
    calibration_refs = load_calibration(num_screens)
    dwell_seconds = config.get("dwell_seconds", 0.05)
    smooth_alpha = config.get("smooth_alpha", 0.52)
    selector = ScreenSelector(dwell_seconds, smooth_alpha, num_screens, calibration_refs)
    move_cursor(*scaled_pos(0))
    frame_dt = 1.0 / fps_limit if fps_limit else 0.05
    last_time = time.perf_counter()
    frame_timestamp_ms = 0
    ms_per_frame = max(1, int(1000 / fps_limit)) if fps_limit else 33
    last_screen_index: int | None = None
    labels = ["Left", "Right"] if num_screens == 2 else ["Left", "Middle", "Right"]
    # Reminders
    session_start = time.perf_counter()
    last_break_reminder = session_start
    last_eye_strain_reminder = session_start
    break_reminder_interval = config.get("break_reminder_interval", 1200)
    eye_strain_reminder_interval = config.get("eye_strain_reminder_interval", 600)
    enable_break_reminder = config.get("enable_break_reminder", True)
    enable_eye_strain_reminder = config.get("enable_eye_strain_reminder", True)
    reminder_overlay_text: str | None = None
    reminder_overlay_until = 0.0
    REMINDER_DISPLAY_SEC = 8.0
    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            dt = now - last_time
            if dt < frame_dt:
                time.sleep(min(frame_dt - dt, 0.05))
                continue
            last_time = time.perf_counter()
            frame_timestamp_ms += ms_per_frame
            config = get_config()
            head_sensitivity = config.get("head_sensitivity", 1.15)
            selector.smooth_alpha = config.get("smooth_alpha", 0.52)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            norm_x = get_head_turn_norm_x(frame, face_mesh, head_sensitivity, frame_timestamp_ms)
            screen_index = selector.update(norm_x, dt)
            if screen_index != last_screen_index and screen_index < len(cursor_positions):
                move_cursor(*scaled_pos(screen_index))
            last_screen_index = screen_index
            now_sec = time.perf_counter()
            if reminder_overlay_text and now_sec > reminder_overlay_until:
                reminder_overlay_text = None
            if not reminder_overlay_text:
                if enable_eye_strain_reminder and (now_sec - last_eye_strain_reminder) >= eye_strain_reminder_interval:
                    reminder_overlay_text = "Eye strain reminder: Look away from the screen for 20 seconds"
                    reminder_overlay_until = now_sec + REMINDER_DISPLAY_SEC
                    last_eye_strain_reminder = now_sec
                elif enable_break_reminder and (now_sec - last_break_reminder) >= break_reminder_interval:
                    reminder_overlay_text = "Take a break! Step away from the screen for a few minutes"
                    reminder_overlay_until = now_sec + REMINDER_DISPLAY_SEC
                    last_break_reminder = now_sec
            if frame_queue is not None:
                try:
                    w, h = frame.shape[1], frame.shape[0]
                    n = len(cursor_positions)
                    bar_y, bar_h, bar_margin = h - 36, 32, 12
                    cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (38, 40, 48), -1)
                    cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (70, 75, 90), 1)
                    seg_w = (w - 2 * bar_margin) // n
                    cx0 = bar_margin + screen_index * seg_w
                    cv2.rectangle(frame, (cx0 + 2, bar_y + 2), (cx0 + seg_w - 2, bar_y + bar_h - 2), (70, 140, 220), -1)
                    for i in range(n):
                        x = bar_margin + seg_w * i + seg_w // 2
                        label = labels[i]
                        text_w = len(label) * 9
                        text_color = (255, 255, 255) if i == screen_index else (150, 150, 160)
                        cv2.putText(frame, label, (x - text_w // 2, bar_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                    status_text = "Screen: " + labels[screen_index]
                    cv2.rectangle(frame, (10, 10), (180, 45), (38, 40, 48), -1)
                    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 170, 240), 2)
                    if reminder_overlay_text:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (20, 25, 35), -1)
                        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                        _draw_text_panel(frame, [reminder_overlay_text], y_start=h // 2 - 50, line_height=50, font_scale=0.9, thickness=2, title="Reminder")
                    try:
                        frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            frame_queue.put_nowait(frame.copy())
                        except queue.Full:
                            pass
                except Exception:
                    pass
    finally:
        cap.release()


class ScreenSelector:
    def __init__(self, dwell_seconds: float, smooth_alpha: float, num_screens: int, calibration_refs: list[float] | None = None):
        self.dwell_seconds = dwell_seconds
        self.smooth_alpha = smooth_alpha
        self.num_screens = num_screens
        self._refs = calibration_refs if calibration_refs and len(calibration_refs) == num_screens else None
        self._smooth_x: float | None = None
        self._current_screen = 0
        self._target_screen = 0
        self._since_change = 0.0

    def _screen_from_x(self, x: float) -> int:
        if self._refs is not None:
            best = 0
            best_d = abs(x - self._refs[0])
            for i in range(1, len(self._refs)):
                d = abs(x - self._refs[i])
                if d < best_d:
                    best_d = d
                    best = i
            return best
        if self.num_screens == 2:
            return 0 if x < 0.5 else 1
        return min(int(x * self.num_screens), self.num_screens - 1)

    def update(self, norm_x: float | None, dt: float) -> int:
        if norm_x is None:
            return self._current_screen
        if self._smooth_x is None:
            self._smooth_x = norm_x
        else:
            self._smooth_x = self.smooth_alpha * self._smooth_x + (1.0 - self.smooth_alpha) * norm_x
        self._target_screen = self._screen_from_x(self._smooth_x)
        if self._target_screen != self._current_screen:
            self._since_change += dt
            if self._since_change >= self.dwell_seconds:
                self._current_screen = self._target_screen
                self._since_change = 0.0
        else:
            self._since_change = 0.0
        return self._current_screen


def run(config_path: Path, no_preview: bool, force_calibrate: bool = False) -> None:
    import cv2
    config = load_config(config_path)
    camera_index = config.get("camera_index", 0)
    frame_w = config.get("frame_width", 640)
    frame_h = config.get("frame_height", 480)
    fps_limit = config.get("fps_limit", 24)
    dwell_seconds = config.get("dwell_seconds", 0.05)
    smooth_alpha = config.get("smooth_alpha", 0.52)
    head_sensitivity = config.get("head_sensitivity", 1.15)
    screen_order = config.get("screen_order", "auto")
    cursor_scale = config.get("cursor_scale", 1.0)

    monitors = get_ordered_monitors(screen_order)
    if len(monitors) < 2:
        print("Need at least 2 monitors. Found:", len(monitors), file=sys.stderr)
        sys.exit(1)

    num_screens = len(monitors)
    cursor_positions = [center_of_region(*m) for m in monitors]

    def scaled_pos(i: int) -> tuple[int, int]:
        cx, cy = cursor_positions[i]
        return (int(cx * cursor_scale), int(cy * cursor_scale))

    if not shutil.which("xdotool"):
        print("xdotool is required for cursor movement. Install it, e.g.:", file=sys.stderr)
        print("  Fedora/RHEL: sudo dnf install xdotool", file=sys.stderr)
        print("  Debian/Ubuntu: sudo apt install xdotool", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    face_mesh = init_face_mesh()

    calibration_refs: list[float] | None = None
    if force_calibrate:
        print("Calibration: follow the on-screen steps.")
        calibration_refs = run_calibration(cap, face_mesh, config, cursor_positions, scaled_pos, num_screens)
        face_mesh = init_face_mesh()  # fresh instance so main loop timestamps start from 0
        print("Calibration saved. Starting tracker...")
    else:
        choice = show_calibration_choice(cap, num_screens)
        if choice == "calibrate":
            calibration_refs = run_calibration(cap, face_mesh, config, cursor_positions, scaled_pos, num_screens)
            face_mesh = init_face_mesh()  # fresh instance so main loop timestamps start from 0
            print("Calibration saved. Starting tracker...")
        elif choice == "saved":
            calibration_refs = load_calibration(num_screens)
            if calibration_refs is None:
                calibration_refs = None
                print("Using defaults.")
            else:
                print("Using saved calibration for mapping.")
        else:
            calibration_refs = None
            print("Using default screen mapping.")

    move_cursor(*scaled_pos(0))
    selector = ScreenSelector(dwell_seconds, smooth_alpha, num_screens, calibration_refs)

    frame_dt = 1.0 / fps_limit if fps_limit else 0.05
    last_time = time.perf_counter()
    tracker_window = "ScreenGaze - Tracker"
    cv2.namedWindow(tracker_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(tracker_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(tracker_window, 640, 480)
    _center_window(tracker_window, 640, 480)
    last_screen_index: int | None = None
    frame_timestamp_ms = 0
    ms_per_frame = max(1, int(1000 / fps_limit)) if fps_limit else 33

    # Break and eye strain reminders (more frequent eye strain reminders)
    session_start = time.perf_counter()
    last_break_reminder = session_start
    last_eye_strain_reminder = session_start
    break_reminder_interval = config.get("break_reminder_interval", 1200)  # 20 min default
    eye_strain_reminder_interval = config.get("eye_strain_reminder_interval", 600)  # 10 min default
    enable_break_reminder = config.get("enable_break_reminder", True)
    enable_eye_strain_reminder = config.get("enable_eye_strain_reminder", True)
    reminder_overlay_text: str | None = None
    reminder_overlay_until = 0.0
    REMINDER_DISPLAY_SEC = 8.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_time
            if dt < frame_dt:
                time.sleep(frame_dt - dt)
            last_time = time.perf_counter()
            frame_timestamp_ms += ms_per_frame

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            norm_x = get_head_turn_norm_x(frame, face_mesh, head_sensitivity, frame_timestamp_ms)
            screen_index = selector.update(norm_x, dt)

            if screen_index != last_screen_index and screen_index < len(cursor_positions):
                move_cursor(*scaled_pos(screen_index))
            last_screen_index = screen_index

            # Check break and eye strain reminders (eye strain more frequent)
            now_sec = time.perf_counter()
            if reminder_overlay_text and now_sec > reminder_overlay_until:
                reminder_overlay_text = None
            if not reminder_overlay_text:
                if enable_eye_strain_reminder and (now_sec - last_eye_strain_reminder) >= eye_strain_reminder_interval:
                    reminder_overlay_text = "Eye strain reminder: Look away from the screen for 20 seconds"
                    reminder_overlay_until = now_sec + REMINDER_DISPLAY_SEC
                    last_eye_strain_reminder = now_sec
                    if no_preview:
                        print("[ScreenGaze] Eye strain reminder: look away for 20 seconds.", flush=True)
                elif enable_break_reminder and (now_sec - last_break_reminder) >= break_reminder_interval:
                    reminder_overlay_text = "Take a break! Step away from the screen for a few minutes"
                    reminder_overlay_until = now_sec + REMINDER_DISPLAY_SEC
                    last_break_reminder = now_sec
                    if no_preview:
                        print("[ScreenGaze] Break reminder: step away for a few minutes.", flush=True)

            if not no_preview:
                w, h = frame.shape[1], frame.shape[0]
                n = len(cursor_positions)
                
                # Modern status bar at bottom
                bar_y, bar_h, bar_margin = h - 36, 32, 12
                # Bar background with rounded corners look
                cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (38, 40, 48), -1)
                cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (70, 75, 90), 1)
                
                # Current screen highlight with gradient effect
                seg_w = (w - 2 * bar_margin) // n
                cx0 = bar_margin + screen_index * seg_w
                # Highlight box
                cv2.rectangle(frame, (cx0 + 2, bar_y + 2), (cx0 + seg_w - 2, bar_y + bar_h - 2), (70, 140, 220), -1)
                cv2.rectangle(frame, (cx0 + 2, bar_y + 2), (cx0 + seg_w - 2, bar_y + bar_h - 2), (100, 170, 240), 1)
                
                # Screen labels
                labels = ["Left", "Right"] if n == 2 else ["Left", "Middle", "Right"]
                for i in range(n):
                    x = bar_margin + seg_w * i + seg_w // 2
                    label = labels[i]
                    # Center text
                    text_w = len(label) * 9
                    text_color = (255, 255, 255) if i == screen_index else (150, 150, 160)
                    cv2.putText(frame, label, (x - text_w // 2, bar_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                
                # Moving dot = where your head is (smooth)
                if norm_x is not None and selector._smooth_x is not None:
                    dot_x = bar_margin + int((w - 2 * bar_margin) * selector._smooth_x)
                    dot_x = max(bar_margin + 8, min(w - bar_margin - 8, dot_x))
                    # Glow effect
                    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 8, (0, 200, 255), -1)
                    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 8, (100, 220, 255), 2)
                    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 4, (255, 255, 255), -1)
                # Top status bar (use ASCII to avoid ??? for special chars)
                cv2.rectangle(frame, (10, 10), (180, 45), (38, 40, 48), -1)
                cv2.rectangle(frame, (10, 10), (180, 45), (70, 75, 90), 1)
                status_text = "Screen: " + labels[screen_index]
                cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 170, 240), 2)

                # Reminder overlay (break / eye strain)
                if reminder_overlay_text:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (20, 25, 35), -1)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    _draw_text_panel(frame, [reminder_overlay_text], y_start=h // 2 - 50, line_height=50, font_scale=0.9, thickness=2, title="Reminder")

                # Quit hint in corner
                cv2.putText(frame, "q = quit", (w - 90, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 150), 1)
                cv2.imshow(tracker_window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def list_monitors() -> None:
    """Print monitor layout for calibration."""
    monitors = list(get_monitors())
    by_x = sorted(monitors, key=lambda m: m.x)
    print("Monitors (sorted by X):")
    for i, m in enumerate(by_x):
        cx, cy = m.x + m.width // 2, m.y + m.height // 2
        print(f"  Index {i}: x={m.x} y={m.y} {m.width}x{m.height}  center=({cx},{cy})  name={getattr(m,'name','?')}")
    ordered = get_ordered_monitors("auto")
    print("\nAuto order (left=0, middle=1, right=2):")
    for i, (x, y, w, h) in enumerate(ordered):
        print(f"  Screen {i}: ({x}, {y}) {w}x{h}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-screen face tracker: cursor follows your face.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--no-preview", action="store_true", help="Run without camera preview window")
    parser.add_argument("--list-monitors", action="store_true", help="Print monitor layout and exit")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration wizard and save (then start tracker)")
    args = parser.parse_args()
    if args.list_monitors:
        list_monitors()
        return
    run(args.config, args.no_preview, args.calibrate)


if __name__ == "__main__":
    main()
