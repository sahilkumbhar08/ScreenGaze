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
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
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


def is_wearing_glasses(frame, lm_list) -> bool:
    """
    Detect if user is wearing glasses by analyzing the eye region in the frame.
    Uses edge detection to look for glasses frames (horizontal lines across eyes).
    """
    try:
        import cv2
        import numpy as np
        
        h, w = frame.shape[:2]
        
        # Get eye region landmarks
        left_eye_outer = lm_list[LEFT_EYE_OUTER]
        left_eye_inner = lm_list[LEFT_EYE_INNER]
        right_eye_outer = lm_list[RIGHT_EYE_OUTER]
        right_eye_inner = lm_list[RIGHT_EYE_INNER]
        
        # Convert normalized coordinates to pixel coordinates
        left_eye_x = int(left_eye_outer.x * w)
        left_eye_y = int(left_eye_outer.y * h)
        right_eye_x = int(right_eye_inner.x * w)
        right_eye_y = int(right_eye_inner.y * h)
        
        # Define eye region (covering both eyes with some padding)
        eye_y_min = min(left_eye_y, right_eye_y) - 40
        eye_y_max = max(left_eye_y, right_eye_y) + 40
        eye_x_min = left_eye_x - 40
        eye_x_max = int(right_eye_inner.x * w) + 40
        
        # Ensure bounds
        eye_y_min = max(0, eye_y_min)
        eye_y_max = min(h, eye_y_max)
        eye_x_min = max(0, eye_x_min)
        eye_x_max = min(w, eye_x_max)
        
        if eye_x_max <= eye_x_min or eye_y_max <= eye_y_min:
            return False
        
        # Extract eye region
        eye_region = frame[eye_y_min:eye_y_max, eye_x_min:eye_x_max]
        
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Look for horizontal lines (glasses frames typically create horizontal edges)
        # Use morphological operations to enhance horizontal lines
        kernel_horizontal = np.ones((1, 15), np.uint8)
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_horizontal, iterations=2)
        
        # Count horizontal line pixels
        horizontal_pixels = np.sum(horizontal_lines > 0)
        total_pixels = horizontal_lines.shape[0] * horizontal_lines.shape[1]
        horizontal_ratio = horizontal_pixels / total_pixels
        
        # Also check for strong vertical edges on sides (temples of glasses)
        kernel_vertical = np.ones((15, 1), np.uint8)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_vertical, iterations=2)
        vertical_pixels = np.sum(vertical_lines > 0)
        vertical_ratio = vertical_pixels / total_pixels
        
        # Check for frame-like structures (rectangular shapes)
        # Use contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_like_contours = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = float(cw)/ch if ch > 0 else 0
            # Glasses frames are typically wider than tall
            if aspect_ratio > 2.0 and cw > 30 and ch > 5:
                frame_like_contours += 1
        
        # Detection logic: combination of horizontal lines and frame-like contours
        # Glasses typically produce strong horizontal edges (top/bottom of frames)
        has_horizontal_frame = horizontal_ratio > 0.02  # At least 2% horizontal edges
        has_frame_structure = frame_like_contours >= 1  # At least one frame-like contour
        
        # Additional check: intensity variation in eye region
        # Glasses create distinct bright/dark patterns
        std_dev = np.std(gray)
        has_texture = std_dev > 30  # Significant texture variation
        
        wearing_glasses = bool((has_horizontal_frame and has_frame_structure) or \
                         (horizontal_ratio > 0.03 and has_texture))
        
        return wearing_glasses
        
    except Exception as e:
        # If detection fails, default to not wearing glasses
        return False


def send_notification(message: str) -> None:
    """Send desktop notification."""
    try:
        # Try notify-send first (most Linux desktops)
        subprocess.run(
            ["notify-send", "ScreenGaze", message, "-u", "normal", "-t", "5000"],
            capture_output=True,
            timeout=5,
        )
    except:
        # Fallback to zenity
        try:
            subprocess.run(
                ["zenity", "--info", "--text", message, "--title", "ScreenGaze"],
                capture_output=True,
                timeout=5,
            )
        except:
            pass


class ScreenSelector:
    def __init__(self, dwell_seconds: float, smooth_alpha: float, num_screens: int):
        self.dwell_seconds = dwell_seconds
        self.smooth_alpha = smooth_alpha
        self.num_screens = num_screens
        self._smooth_x: float | None = None
        self._current_screen = 0
        self._target_screen = 0
        self._since_change = 0.0

    def update(self, norm_x: float | None, dt: float) -> int:
        if norm_x is None:
            return self._current_screen
        if self._smooth_x is None:
            self._smooth_x = norm_x
        else:
            self._smooth_x = self.smooth_alpha * self._smooth_x + (1.0 - self.smooth_alpha) * norm_x
        # Map to screen index
        t = self._smooth_x
        if self.num_screens == 2:
            self._target_screen = 0 if t < 0.5 else 1
        else:
            self._target_screen = min(
                int(t * self.num_screens),
                self.num_screens - 1,
            )
        # Very short dwell so a neck turn switches quickly
        if self._target_screen != self._current_screen:
            self._since_change += dt
            if self._since_change >= self.dwell_seconds:
                self._current_screen = self._target_screen
                self._since_change = 0.0
        else:
            self._since_change = 0.0
        return self._current_screen


def run(config_path: Path, no_preview: bool) -> None:
    import cv2
    config = load_config(config_path)
    camera_index = config.get("camera_index", 0)
    frame_w = config.get("frame_width", 640)
    frame_h = config.get("frame_height", 480)
    fps_limit = config.get("fps_limit", 15)
    dwell_seconds = config.get("dwell_seconds", 0.06)
    smooth_alpha = config.get("smooth_alpha", 0.35)
    head_sensitivity = config.get("head_sensitivity", 1.2)
    screen_order = config.get("screen_order", "auto")
    cursor_scale = config.get("cursor_scale", 1.0)

    monitors = get_ordered_monitors(screen_order)
    if len(monitors) < 2:
        print("Need at least 2 monitors. Found:", len(monitors), file=sys.stderr)
        sys.exit(1)

    print("Monitor order (left → middle → right):")
    for i, (x, y, w, h) in enumerate(monitors):
        cx, cy = center_of_region(x, y, w, h)
        print(f"  Screen {i}: ({x}, {y}) size {w}x{h}  → cursor target ({cx}, {cy})")

    # Build list of (cx, cy) per screen for cursor target
    cursor_positions = [center_of_region(*m) for m in monitors]

    def scaled_pos(i: int) -> tuple[int, int]:
        cx, cy = cursor_positions[i]
        return (int(cx * cursor_scale), int(cy * cursor_scale))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    if not shutil.which("xdotool"):
        print("xdotool is required for cursor movement. Install it, e.g.:", file=sys.stderr)
        print("  Fedora/RHEL: sudo dnf install xdotool", file=sys.stderr)
        print("  Debian/Ubuntu: sudo apt install xdotool", file=sys.stderr)
        sys.exit(1)
    move_cursor(*scaled_pos(0))
    face_mesh = init_face_mesh()
    selector = ScreenSelector(dwell_seconds, smooth_alpha, len(cursor_positions))

    frame_dt = 1.0 / fps_limit if fps_limit else 0.05
    last_time = time.perf_counter()
    window_name = "Face tracker (q=quit)"
    last_screen_index: int | None = None
    frame_timestamp_ms = 0
    ms_per_frame = max(1, int(1000 / fps_limit)) if fps_limit else 33
    
    # Glasses reminder timer
    glasses_reminder_enabled = config.get("enable_glasses_reminder", True)
    GLASSES_REMINDER_INTERVAL = config.get("glasses_reminder_interval", 120)  # seconds
    glasses_check_timer = 0.0
    last_glasses_notification = 0.0
    wearing_glasses = False
    glasses_detection_frames = 0
    frames_without_glasses = 0
    FRAMES_TO_CONFIRM = 10  # Need 10 consecutive frames to confirm glasses status

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

            import cv2
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_mesh.detect_for_video(mp_image, frame_timestamp_ms)
            
            # Get norm_x from the same result
            norm_x = None
            if result.face_landmarks:
                lm_list = result.face_landmarks[0]
                nose = lm_list[NOSE_TIP_IDX]
                xs = [p.x for p in lm_list]
                left_x, right_x = min(xs), max(xs)
                face_center_x = (left_x + right_x) * 0.5
                face_half_width = max((right_x - left_x) * 0.5, 1e-6)
                yaw_proxy = (nose.x - face_center_x) / face_half_width
                yaw_proxy = max(-1.0, min(1.0, yaw_proxy * head_sensitivity))
                norm_x = 0.5 - 0.5 * yaw_proxy
            
            # Glasses detection using the same result
            face_detected = False
            if result.face_landmarks:
                face_detected = True
                lm_list = result.face_landmarks[0]
                currently_wearing = is_wearing_glasses(frame, lm_list)
                
                if currently_wearing:
                    glasses_detection_frames += 1
                    frames_without_glasses = 0
                else:
                    frames_without_glasses += 1
                    glasses_detection_frames = 0
                
                # Confirm glasses status after FRAMES_TO_CONFIRM consecutive frames
                if glasses_detection_frames >= FRAMES_TO_CONFIRM:
                    wearing_glasses = True
                elif frames_without_glasses >= FRAMES_TO_CONFIRM:
                    wearing_glasses = False
            else:
                # No face detected, reset counters
                glasses_detection_frames = 0
                frames_without_glasses = 0
            
            # Check if we need to send glasses reminder
            if glasses_reminder_enabled:
                glasses_check_timer += dt
                time_remaining = GLASSES_REMINDER_INTERVAL - glasses_check_timer
                
                # Debug output every 10 seconds
                if int(glasses_check_timer) % 10 == 0 and int(glasses_check_timer) > 0:
                    print(f"[DEBUG] Glasses check timer: {glasses_check_timer:.1f}s / {GLASSES_REMINDER_INTERVAL}s")
                    print(f"[DEBUG] Wearing glasses: {wearing_glasses}, Face detected: {face_detected}")
                
                if glasses_check_timer >= GLASSES_REMINDER_INTERVAL:
                    print(f"[DEBUG] Timer triggered! Wearing glasses: {wearing_glasses}")
                    glasses_check_timer = 0.0
                    if not wearing_glasses:
                        print("[DEBUG] Sending notification...")
                        send_notification("👓 Time to wear your computer glasses! Protect your eyes.")
                        last_glasses_notification = now
                        print("[DEBUG] Notification sent!")
            
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
                labels = ["Left", "Middle", "Right"][:n]
                
                # Glasses status indicator and timer
                glasses_status = "👓 ON" if wearing_glasses else "👓 OFF"
                glasses_color = (0, 255, 0) if wearing_glasses else (0, 0, 255)
                cv2.putText(
                    frame, glasses_status, (w - 100, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, glasses_color, 2,
                )
                
                # Show reminder timer
                if glasses_reminder_enabled:
                    remaining = max(0, GLASSES_REMINDER_INTERVAL - glasses_check_timer)
                    timer_text = f"⏰ {int(remaining)}s"
                    timer_color = (0, 255, 255) if remaining > 30 else (0, 165, 255) if remaining > 10 else (0, 0, 255)
                    cv2.putText(
                        frame, timer_text, (w - 200, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2,
                    )
                
                cv2.putText(
                    frame, "Screen: %s  (q=quit)" % labels[screen_index], (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                )
                
                cv2.imshow(window_name, frame)
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
    args = parser.parse_args()
    if args.list_monitors:
        list_monitors()
        return
    run(args.config, args.no_preview)


if __name__ == "__main__":
    main()
