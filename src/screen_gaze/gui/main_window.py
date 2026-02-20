"""Main application window for ScreenGaze with modern dark UI."""
from __future__ import annotations
import json
import shutil
import queue
import threading
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, Frame, Label, Scale, StringVar, Tk
from tkinter import Button, Checkbutton, messagebox
import cv2
from PIL import Image, ImageTk
from screen_gaze.gui.calibration_dialog import CalibrationDialog
from screen_gaze.core.face_tracker import run_tracking_loop
from screeninfo import get_monitors
from screen_gaze.gui.theme import (
    BG_CARD,
    BG_DARK,
    BG_INPUT,
    BORDER,
    BUTTON_START,
    BUTTON_STOP,
    FONT_BODY,
    FONT_HEADING,
    FONT_SMALL,
    FONT_STATUS,
    FONT_TITLE,
    PAD_LG,
    PAD_MD,
    PAD_SM,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    SLIDER_ACTIVE,
    SLIDER_TROUGH,
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)


class MainWindow:
    """Main ScreenGaze control window: calibration, start/stop, preview, sensitivity, smoothing."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._config: dict = {}
        self._live_config: dict = {}
        self._config_lock = threading.Lock()
        self._frame_queue: queue.Queue | None = None
        self._stop_event = threading.Event()
        self._tracking_thread: threading.Thread | None = None
        self._preview_job: str | None = None
        self._photo = None  # keep reference so image is not garbage-collected
        self._root = Tk()
        self._root.title("ScreenGaze")
        self._root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._root.configure(bg=BG_DARK)
        self._root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._load_config()
        self._build_ui()
        self._root.after(100, self._poll_preview)

    def _load_config(self) -> None:
        try:
            with open(self.config_path) as f:
                self._config = json.load(f)
        except Exception:
            self._config = {
                "head_sensitivity": 1.15,
                "smooth_alpha": 0.52,
                "frame_width": 640,
                "frame_height": 480,
                "fps_limit": 24,
                "camera_index": 0,
                "screen_order": "auto",
                "cursor_scale": 1.0,
                "dwell_seconds": 0.05,
                "break_reminder_interval": 1200,
                "eye_strain_reminder_interval": 600,
                "enable_break_reminder": True,
                "enable_eye_strain_reminder": True,
            }
        self._sensitivity_var = DoubleVar(value=self._config.get("head_sensitivity", 1.15))
        self._smoothing_var = BooleanVar(value=self._config.get("smooth_alpha", 0.52) >= 0.3)
        self._status_var = StringVar(value="Not Active")
        self._sync_live_config()

    def _sync_live_config(self) -> None:
        """Update thread-safe config copy from current vars (call from main thread)."""
        with self._config_lock:
            self._live_config = dict(self._config)
            self._live_config["head_sensitivity"] = round(self._sensitivity_var.get(), 2)
            self._live_config["smooth_alpha"] = 0.52 if self._smoothing_var.get() else 0.0

    def _save_config(self) -> None:
        self._config["head_sensitivity"] = round(self._sensitivity_var.get(), 2)
        self._config["smooth_alpha"] = 0.52 if self._smoothing_var.get() else 0.0
        self._sync_live_config()
        try:
            with open(self.config_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception:
            pass

    def _get_config_for_thread(self) -> dict:
        """Thread-safe: return a copy of config for the tracking thread (no Tk access)."""
        with self._config_lock:
            return dict(self._live_config)

    def _build_ui(self) -> None:
        content = Frame(self._root, bg=BG_DARK, padx=PAD_LG, pady=PAD_LG)
        content.pack(fill="both", expand=True)

        # Title
        title = Label(
            content,
            text="ScreenGaze",
            font=FONT_TITLE,
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
        )
        title.pack(pady=(0, PAD_MD))

        # Preview card
        preview_card = Frame(content, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        preview_card.pack(fill="x", pady=(0, PAD_MD))
        preview_inner = Frame(preview_card, bg=BG_CARD, padx=PAD_SM, pady=PAD_SM)
        preview_inner.pack()
        # Use a fixed-size frame so the image has space; label will size to image when set
        preview_label = Label(
            preview_inner,
            text="Camera preview\n(Start tracking to see feed)",
            font=FONT_BODY,
            fg=TEXT_SECONDARY,
            bg=BG_INPUT,
        )
        preview_label.pack(padx=PAD_SM, pady=PAD_SM)
        self._preview_label = preview_label

        # Calibration hint
        Label(
            content,
            text="Calibrate before first use for accurate screen mapping.",
            font=FONT_SMALL,
            fg=TEXT_SECONDARY,
            bg=BG_DARK,
        ).pack(pady=(0, PAD_SM))

        # Status
        status_frame = Frame(content, bg=BG_DARK)
        status_frame.pack(fill="x", pady=(0, PAD_MD))
        Label(status_frame, text="Tracking status:", font=FONT_HEADING, fg=TEXT_SECONDARY, bg=BG_DARK).pack(side="left")
        self._status_label = Label(
            status_frame,
            textvariable=self._status_var,
            font=FONT_STATUS,
            fg=STATUS_INACTIVE,
            bg=BG_DARK,
        )
        self._status_label.pack(side="left", padx=(PAD_SM, 0))

        # Buttons (centered, large)
        btn_outer = Frame(content, bg=BG_DARK)
        btn_outer.pack(fill="x", pady=PAD_MD)
        btn_frame = Frame(btn_outer, bg=BG_DARK)
        btn_frame.pack(expand=True)
        self._btn_calibrate = Button(
            btn_frame,
            text="Calibrate",
            font=FONT_BODY,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            activebackground=BG_INPUT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            height=2,
            width=14,
            command=self._on_calibrate,
        )
        self._btn_calibrate.pack(side="left", padx=PAD_SM)
        self._btn_start_stop = Button(
            btn_frame,
            text="Start Tracking",
            font=FONT_BODY,
            bg=BUTTON_START,
            fg="white",
            activebackground=BUTTON_START,
            activeforeground="white",
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            height=2,
            width=14,
            command=self._on_start_stop,
        )
        self._btn_start_stop.pack(side="left", padx=PAD_SM)
        self._style_buttons()

        # Sensitivity slider
        sens_frame = Frame(content, bg=BG_DARK)
        sens_frame.pack(fill="x", pady=PAD_SM)
        Label(sens_frame, text="Sensitivity", font=FONT_HEADING, fg=TEXT_PRIMARY, bg=BG_DARK).pack(anchor="w")
        self._sensitivity_scale = Scale(
            sens_frame,
            from_=0.5,
            to=2.0,
            resolution=0.05,
            orient="horizontal",
            variable=self._sensitivity_var,
            bg=BG_DARK,
            fg=TEXT_PRIMARY,
            troughcolor=SLIDER_TROUGH,
            activebackground=SLIDER_ACTIVE,
            highlightthickness=0,
            length=280,
            command=self._on_sensitivity_change,
        )
        self._sensitivity_scale.pack(fill="x", pady=(PAD_SM, 0))
        Label(sens_frame, text="Low  -  High", font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_DARK).pack(anchor="w")

        # Smoothing toggle
        smooth_frame = Frame(content, bg=BG_DARK)
        smooth_frame.pack(fill="x", pady=PAD_MD)
        self._smoothing_cb = Checkbutton(
            smooth_frame,
            text="Smooth cursor movement",
            variable=self._smoothing_var,
            font=FONT_BODY,
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
            activebackground=BG_DARK,
            activeforeground=TEXT_PRIMARY,
            selectcolor=BG_INPUT,
            command=self._on_smoothing_change,
        )
        self._smoothing_cb.pack(anchor="w")

    def _style_buttons(self) -> None:
        for btn in (self._btn_calibrate, self._btn_start_stop):
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=BG_INPUT if b == self._btn_calibrate else "#40d86a"))
            btn.bind("<Leave>", lambda e, b=btn: self._update_button_colors())

    def _update_button_colors(self) -> None:
        if self._status_var.get() == "Active":
            self._btn_start_stop.configure(bg=BUTTON_STOP, text="Stop Tracking")
        else:
            self._btn_start_stop.configure(bg=BUTTON_START, text="Start Tracking")
        self._btn_calibrate.configure(bg=BG_CARD)

    def _on_sensitivity_change(self, _value: str) -> None:
        self._sync_live_config()
        self._save_config()

    def _on_smoothing_change(self) -> None:
        self._sync_live_config()
        self._save_config()

    def _on_calibrate(self) -> None:
        dialog = CalibrationDialog(self._root, self.config_path)
        self._root.wait_window(dialog.top)
        if dialog.result == "calibrated":
            messagebox.showinfo("Calibration", "Calibration completed. You can start tracking.")
        elif dialog.result == "saved":
            messagebox.showinfo("Calibration", "Using saved calibration.")
        elif dialog.result == "defaults":
            pass

    def _on_start_stop(self) -> None:
        if self._status_var.get() == "Active":
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self) -> None:
        if not self._check_requirements():
            return
        self._sync_live_config()
        self._frame_queue = queue.Queue(maxsize=6)
        self._stop_event.clear()
        self._tracking_thread = threading.Thread(
            target=run_tracking_loop,
            args=(self.config_path, self._frame_queue, self._stop_event, self._get_config_for_thread),
            daemon=True,
        )
        self._tracking_thread.start()
        self._status_var.set("Active")
        self._status_label.configure(fg=STATUS_ACTIVE)
        self._btn_start_stop.configure(bg=BUTTON_STOP, text="Stop Tracking")

    def _stop_tracking(self) -> None:
        self._stop_event.set()
        if self._tracking_thread is not None:
            self._tracking_thread.join(timeout=3.0)
            self._tracking_thread = None
        self._frame_queue = None
        self._status_var.set("Not Active")
        self._status_label.configure(fg=STATUS_INACTIVE)
        self._btn_start_stop.configure(bg=BUTTON_START, text="Start Tracking")
        self._preview_label.configure(text="Camera preview\n(Start tracking to see feed)", image="")
        self._photo = None

    def _check_requirements(self) -> bool:
        if not shutil.which("xdotool"):
            messagebox.showerror(
                "Error",
                "xdotool is required for cursor movement.\n"
                "Install it, e.g.:\n  Fedora: sudo dnf install xdotool\n  Ubuntu: sudo apt install xdotool",
            )
            return False
        
        if len(list(get_monitors())) < 2:
            messagebox.showerror("Error", "ScreenGaze needs at least 2 monitors.")
            return False
        return True

    def _poll_preview(self) -> None:
        if self._frame_queue is not None:
            try:
                frame = self._frame_queue.get_nowait()
                self._show_frame(frame)
            except queue.Empty:
                pass
        self._preview_job = self._root.after(50, self._poll_preview)

    def _show_frame(self, frame) -> None:
        
        try:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), getattr(Image, "Resampling", Image).LANCZOS)
            self._photo = ImageTk.PhotoImage(image=img)
            self._preview_label.configure(image=self._photo, text="")
        except Exception:
            pass

    def _on_closing(self) -> None:
        if self._status_var.get() == "Active":
            self._stop_tracking()
        if self._preview_job:
            self._root.after_cancel(self._preview_job)
        self._save_config()
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()
