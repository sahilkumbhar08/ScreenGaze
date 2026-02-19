# ScreenGaze

Move your cursor to the screen you're looking at. Uses your webcam and head pose to switch between monitors—no need to drag the mouse. All processing is local; no video is sent over the network.

---

## Features

### Gaze tracking
- **Head-based screen switching** — Look at a monitor; the cursor moves there.
- **Multi-monitor** — Works with 2 or 3 screens (side-by-side or with a lower center display).
- **Calibration** — One-time setup maps your face position to each screen for accurate mapping. Options: calibrate now, use saved calibration, or use defaults.

### GUI (default)
- **Modern dark theme** — Clean layout, clear typography, teal accents (same in main window and calibration).
- **Main window** — Start/Stop tracking, live webcam preview, sensitivity slider, smoothing toggle, tracking status.
- **Calibration screen** — Same theme: choose “Calibrate now”, “Use saved calibration”, or “Use defaults”. “Calibrate now” opens the step-by-step wizard (look at Left/Right/Middle monitor, press **SPACE** to capture; progress bar and completion screen use the same dark theme).
- **Live webcam preview** — Small camera feed with on-screen status when tracking is active.
- **Sensitivity slider** — Adjust head-tracking sensitivity (low to high). Saved to config.
- **Smooth cursor movement** — Toggle to smooth or instant screen switching. Saved to config.
- **Tracking status** — Shows “Active” (green) or “Not Active” (grey).

### Health reminders
- **Break reminders** — Optional reminder to step away (default: every 20 minutes). Shown as an overlay for 8 seconds; can be disabled in config.
- **Eye strain reminders** — More frequent reminder to look away from the screen (default: every 10 minutes). Can be disabled in config.

### CLI mode
- Run with OpenCV camera window instead of the GUI (`--no-gui`).
- Optional: no preview (`--no-preview`), force calibration then start (`--calibrate`), list monitors (`--list-monitors`).

---

## Requirements

- **Python 3.8+**
- **Webcam**
- **Linux with X11** (cursor movement via xdotool)
- **xdotool:**  
  `sudo dnf install xdotool` (Fedora/RHEL) or `sudo apt install xdotool` (Ubuntu/Debian)
- **tkinter** (for GUI):  
  `sudo dnf install python3-tkinter` (Fedora) or `sudo apt install python3-tk` (Ubuntu/Debian)
- **At least 2 monitors** (GUI starts anyway; “Start Tracking” checks this)

---

## Installation

```bash
git clone https://github.com/sahilkumbhar08/ScreenGaze.git
cd ScreenGaze
python3 -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

On first run the app downloads the MediaPipe face model once (~3 MB).

---

## Run

**GUI (default):**
```bash
./run.sh
```
Or:
```bash
python main.py
```

**CLI (OpenCV windows):**
```bash
./run.sh --no-gui
```

**Other options (pass after `./run.sh` or `python main.py`):**
| Option | Description |
|--------|-------------|
| `--no-gui` | Use CLI with OpenCV camera window instead of the GUI. |
| `--list-monitors` | Print monitor layout (left/middle/right order) and exit. |
| `--no-preview` | (CLI only) Run without camera preview window. |
| `--calibrate` | (CLI only) Run calibration wizard, then start tracking. |

**Using run.sh:** Run `./run.sh` from the project directory (or from anywhere; the script `cd`s into its own directory). It activates the virtualenv if `.venv` exists and runs `main.py` with any arguments you pass.

---

## Usage (GUI)

1. **Launch** — Run `./run.sh`. The main window opens (dark theme).
2. **Calibrate (recommended)** — Click **Calibrate**. In the calibration dialog choose:
   - **Calibrate now (recommended)** — Opens the wizard: for each screen, look at that monitor and press **SPACE** to capture. Saves to `calibration.json`.
   - **Use saved calibration** — Use last saved mapping (only if it exists).
   - **Use defaults** — No calibration; fixed left/middle/right zones.
3. **Start tracking** — Click **Start Tracking**. The live webcam preview appears; cursor moves to the screen you look at. Status shows “Active”.
4. **Adjust** — Use the **Sensitivity** slider and **Smooth cursor movement** checkbox; changes are saved to `config.json`.
5. **Stop** — Click **Stop Tracking**. Preview returns to placeholder.

Break and eye strain reminders appear as overlays during tracking when enabled in config.

---

## Configuration

**`config.json`** — Main configuration. The GUI saves **head_sensitivity** and **smooth_alpha** when you change the slider or smoothing toggle.

| Key | Description | Default |
|-----|-------------|--------|
| `camera_index` | Webcam device index | `0` |
| `frame_width`, `frame_height` | Camera resolution | `640`, `480` |
| `fps_limit` | Max FPS for tracking | `24` |
| `head_sensitivity` | Head-turn sensitivity (0.5–2.0 in GUI) | `1.15` |
| `smooth_alpha` | Cursor smoothing (0 = off, 0.52 = on) | `0.52` |
| `dwell_seconds` | Time to dwell before switching screen | `0.05` |
| `screen_order` | Monitor order: `"auto"` or comma-separated indices | `"auto"` |
| `cursor_scale` | Scale factor for cursor position | `1` |
| `break_reminder_interval` | Seconds between break reminders (e.g. 1200 = 20 min) | `1200` |
| `enable_break_reminder` | Show break reminders | `true` |
| `eye_strain_reminder_interval` | Seconds between eye strain reminders (e.g. 600 = 10 min) | `600` |
| `enable_eye_strain_reminder` | Show eye strain reminders | `true` |

**`calibration.json`** — Created by calibration; stores face-to-screen mapping. Do not edit by hand.

---

## Project structure

```
ScreenGaze/
├── main.py              # Entry point (GUI by default; --no-gui for CLI)
├── face_tracker.py      # Core tracking, calibration, reminders
├── config.json          # Configuration (created/edited by app and user)
├── calibration.json     # Calibration data (created by calibration)
├── requirements.txt     # Python dependencies
├── run.sh               # Launcher (venv + main.py)
├── README.md
├── gui/
│   ├── __init__.py
│   ├── theme.py         # Dark theme colors and layout constants
│   ├── main_window.py   # Main application window (preview, buttons, slider, toggle)
│   └── calibration_dialog.py  # Calibration choice dialog (same theme as main window)
└── face_landmarker.task # MediaPipe model (downloaded on first run)
```

`.venv/` and `calibration.json` are local and typically not committed.

---

## Notes

- **Wayland:** Cursor movement may require an X11 session (xdotool uses X11).
- **Single monitor:** The GUI runs, but “Start Tracking” will show an error; at least 2 monitors are required for tracking.
- **Calibration wizard:** The “look at Left/Right/Middle” and “Hold still while capturing…” screens use the same dark theme and teal accents as the main GUI.
- Reminders are shown as an overlay for 8 seconds; in CLI with `--no-preview`, a message is printed instead.
