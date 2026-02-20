"""Calibration screen with same theme as main window: choice and optional wizard."""
from __future__ import annotations

from pathlib import Path
from tkinter import Button, Frame, Label, Toplevel
from screen_gaze.core.face_tracker import run_calibration_standalone
from screen_gaze.core.face_tracker import get_ordered_monitors, load_calibration, load_config
from screen_gaze.gui.theme import (
    BG_CARD,
    BG_DARK,
    BORDER,
    BUTTON_START,
    FONT_BODY,
    FONT_TITLE,
    PAD_LG,
    PAD_MD,
    PAD_SM,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class CalibrationDialog:
    """Modal calibration screen: choose Calibrate now / Use saved / Use defaults. Same dark theme as main GUI."""

    def __init__(self, parent, config_path: Path) -> None:
        self.config_path = config_path
        self.result: str | None = None  # "calibrated" | "saved" | "defaults" | None (cancelled)
        self.top = Toplevel(parent)
        self.top.title("ScreenGaze - Calibration")
        self.top.configure(bg=BG_DARK)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.minsize(420, 380)
        self._build_ui()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self) -> None:
        content = Frame(self.top, bg=BG_DARK, padx=PAD_LG, pady=PAD_LG)
        content.pack(fill="both", expand=True)

        Label(
            content,
            text="Calibration",
            font=FONT_TITLE,
            fg=TEXT_PRIMARY,
            bg=BG_DARK,
        ).pack(pady=(0, PAD_SM))

        Label(
            content,
            text="Choose how to set up screen mapping for gaze tracking.",
            font=FONT_BODY,
            fg=TEXT_SECONDARY,
            bg=BG_DARK,
        ).pack(pady=(0, PAD_MD))

        card = Frame(content, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, PAD_MD))
        inner = Frame(card, bg=BG_CARD, padx=PAD_MD, pady=PAD_MD)
        inner.pack(fill="x")

        btn_calibrate = Button(
            inner,
            text="Calibrate now (recommended)",
            font=FONT_BODY,
            bg=BUTTON_START,
            fg="white",
            activebackground=BUTTON_START,
            activeforeground="white",
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            height=2,
            width=28,
            command=self._on_calibrate_now,
        )
        btn_calibrate.pack(fill="x", pady=PAD_SM)

        try:
            config = load_config(self.config_path)
            monitors = get_ordered_monitors(config.get("screen_order", "auto"))
            num_screens = len(monitors)
            has_saved = load_calibration(num_screens) is not None
        except Exception:
            has_saved = False

        if has_saved:
            btn_saved = Button(
                inner,
                text="Use saved calibration",
                font=FONT_BODY,
                bg=BG_DARK,
                fg=TEXT_PRIMARY,
                activebackground=BG_CARD,
                activeforeground=TEXT_PRIMARY,
                relief="flat",
                highlightthickness=0,
                cursor="hand2",
                height=2,
                width=28,
                command=self._on_use_saved,
            )
            btn_saved.pack(fill="x", pady=PAD_SM)

        btn_defaults = Button(
            inner,
            text="Use defaults (no calibration)",
            font=FONT_BODY,
            bg=BG_DARK,
            fg=TEXT_SECONDARY,
            activebackground=BG_CARD,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            height=2,
            width=28,
            command=self._on_use_defaults,
        )
        btn_defaults.pack(fill="x", pady=PAD_SM)

        Label(
            content,
            text="Calibrate now opens the calibration wizard to map your face to each screen.",
            font=FONT_BODY,
            fg=TEXT_SECONDARY,
            bg=BG_DARK,
        ).pack(pady=(PAD_MD, 0))

    def _on_calibrate_now(self) -> None:
        self.top.grab_release()
        self.top.withdraw()
        try:
            run_calibration_standalone(self.config_path)
            self.result = "calibrated"
        except Exception:
            self.result = None
        self.top.destroy()

    def _on_use_saved(self) -> None:
        self.result = "saved"
        self.top.grab_release()
        self.top.destroy()

    def _on_use_defaults(self) -> None:
        self.result = "defaults"
        self.top.grab_release()
        self.top.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.top.grab_release()
        self.top.destroy()
