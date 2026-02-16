#!/usr/bin/env python3
"""Multi-screen face tracker: look at a screen → cursor moves there."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from screeninfo import get_monitors

CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_CAPTURE_SEC = 1.5
CALIBRATION_MIN_SAMPLES = 20


def _draw_text_panel(frame, lines: list[str], y_start: int = 40, line_height: int = 52, font_scale: float = 1.0, thickness: int = 2):
    """Draw readable text on a dark panel. lines = list of strings."""
    import cv2
    h, w = frame.shape[:2]
    n = len(lines)
    pad = 24
    panel_h = n * line_height + pad * 2
    panel_w = min(w - 80, 700)
    x1, y1 = (w - panel_w) // 2, y_start
    x2, y2 = x1 + panel_w, y1 + panel_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 50), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 100), 2)
    for i, line in enumerate(lines):
        y = y1 + pad + (i + 1) * line_height - 12
        # Shadow for readability
        cv2.putText(frame, line, (x1 + pad + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, line, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


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


def show_calibration_choice(cap, num_screens: int) -> str:
    """Show pop-up: C=Calibrate, S=Use saved, D=Defaults. Returns 'calibrate'|'saved'|'defaults'."""
    import cv2
    window_name = "ScreenGaze — Setup"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    has_saved = load_calibration(num_screens) is not None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        lines = [
            "ScreenGaze  Setup",
            "",
            "  C   Calibrate now (map your face to each screen)",
            "  S   Use saved calibration" if has_saved else "  D  — Defaults (no calibration)",
            "  D   Defaults (no calibration)" if has_saved else "",
        ]
        lines = [s for s in lines if s]
        _draw_text_panel(frame, lines, y_start=60, line_height=48, font_scale=0.95, thickness=2)
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
    scaled_pos: callable,
    num_screens: int,
) -> list[float]:
    import cv2
    sensitivity = config.get("head_sensitivity", 1.15)
    fps = max(10, config.get("fps_limit", 24))
    frame_dt = 1.0 / fps
    labels = ["Left", "Right"] if num_screens == 2 else ["Left", "Middle", "Right"]
    refs: list[float] = []
    window_name = "ScreenGaze — Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
            cv2.rectangle(overlay, (0, 0), (w, h), (35, 35, 45), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            _draw_text_panel(frame, [
                f"Step {step + 1} of {num_screens}",
                f"Look at your {labels[step]} screen.",
                "",
                "Press SPACE when ready to capture.",
            ], y_start=50, line_height=50, font_scale=1.0, thickness=2)
            nx = get_head_turn_norm_x(frame, face_landmarker, sensitivity, frame_ts_ms)
            if nx is None:
                _draw_text_panel(frame, ["Position your face in the camera view"], y_start=h // 2 - 40, line_height=44, font_scale=0.9)
            cv2.putText(frame, "ESC = cancel", (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 190), 2)
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
            cv2.rectangle(overlay, (0, 0), (w, h), (40, 42, 55), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            progress = (c + 1) / capture_frames
            _draw_text_panel(frame, [f"Capturing {labels[step]}...", ""], y_start=60, line_height=46, font_scale=1.05)
            bar_y = 200
            cv2.rectangle(frame, (80, bar_y), (w - 80, bar_y + 24), (50, 50, 60), -1)
            cv2.rectangle(frame, (80, bar_y), (80 + int((w - 160) * progress), bar_y + 24), (60, 140, 220), -1)
            cv2.rectangle(frame, (80, bar_y), (w - 80, bar_y + 24), (90, 90, 110), 1)
            cv2.imshow(window_name, frame)
            cv2.waitKey(max(1, int(frame_dt * 1000)))

        if samples:
            refs.append(sum(samples) / len(samples))
        else:
            refs.append((step + 0.5) / num_screens)

    # Show "Calibration complete" then transition to tracker
    for _ in range(30):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (25, 35, 50), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        _draw_text_panel(frame, ["Calibration complete!", "Starting tracker..."], y_start=h // 2 - 60, line_height=56, font_scale=1.15)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(50) == 27:
            break
    cv2.destroyAllWindows()
    save_calibration(refs, num_screens)
    return refs


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
    tracker_window = "ScreenGaze — Tracker"
    cv2.namedWindow(tracker_window, cv2.WINDOW_NORMAL)
    last_screen_index: int | None = None
    frame_timestamp_ms = 0
    ms_per_frame = max(1, int(1000 / fps_limit)) if fps_limit else 33

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

            if not no_preview:
                w, h = frame.shape[1], frame.shape[0]
                n = len(cursor_positions)
                bar_y, bar_h, bar_margin = h - 28, 22, 8
                cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (45, 45, 45), -1)
                cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (90, 90, 90), 1)
                # Current screen highlight
                seg_w = (w - 2 * bar_margin) // n
                cx0 = bar_margin + screen_index * seg_w
                cv2.rectangle(frame, (cx0, bar_y), (cx0 + seg_w, bar_y + bar_h), (60, 120, 170), -1)
                # Segment dividers
                for i in range(1, n):
                    x = bar_margin + seg_w * i
                    cv2.line(frame, (x, bar_y), (x, bar_y + bar_h), (70, 70, 70), 1)
                # Moving dot = where your head is (smooth)
                if norm_x is not None and selector._smooth_x is not None:
                    dot_x = bar_margin + int((w - 2 * bar_margin) * selector._smooth_x)
                    dot_x = max(bar_margin + 6, min(w - bar_margin - 6, dot_x))
                    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 5, (0, 255, 220), -1)
                    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 5, (255, 255, 255), 1)
                labels = ["Left", "Right"] if n == 2 else ["Left", "Middle", "Right"]
                cv2.rectangle(frame, (8, 8), (220, 42), (45, 45, 55), -1)
                cv2.rectangle(frame, (8, 8), (220, 42), (90, 90, 110), 1)
                cv2.putText(frame, "Screen: %s" % labels[screen_index], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame, "q = quit", (w - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 210), 2)
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
