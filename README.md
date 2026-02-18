# ScreenGaze

Move your cursor to the screen you're looking at. Uses your webcam and head pose to switch between monitors—no need to drag the mouse.

## Features

- **Head-based screen switching** — Look at a monitor; the cursor jumps there
- **Calibration** — One-time setup maps your face position to each screen for accurate mapping
- **Multi-monitor** — Works with 2 or 3 screens (side-by-side or with a lower center display)
- **Privacy** — All processing is local; no video is sent over the network

## Requirements

- Python 3.8+
- Webcam
- Linux with X11
- **xdotool:** `sudo dnf install xdotool` (Fedora) or `sudo apt install xdotool` (Ubuntu/Debian)

## Installation

```bash
git clone https://github.com/sahilkumbhar08/ScreenGaze.git
cd ScreenGaze
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run the app downloads the MediaPipe face model once (~3 MB).

## Run

```bash
./run.sh
```

### User Interface

ScreenGaze features a modern, polished interface with professionally designed windows:

**Setup Window** — Clean, centered dialog with:
- Rounded panels with elegant borders
- Clear typography with keyboard shortcuts highlighted
- Always-on-top for easy access

**Calibration Screens** — Step-by-step guide featuring:
- Progress indicators showing current step
- Visual progress bars during capture
- Clear instructions with status bar

**Tracker Window** — Real-time monitoring with:
- Modern status bar at bottom showing all monitors with highlighted active screen
- Real-time head position indicator with glow effect
- Clean top bar showing current active screen with icon
- Centered window positioned on your primary monitor
- Always-on-top for continuous visibility

### Setup (every run)

A window appears with three options:

| Key | Action |
|-----|--------|
| **C** | **Calibrate now** — Map your face to each screen. For each screen you look at that monitor and press **SPACE** to capture. Saved to `calibration.json` and used for mapping. Recommended for best accuracy. |
| **S** | **Use saved calibration** — Use your last saved mapping (only shown if `calibration.json` exists). |
| **D** | **Defaults** — Run without calibration (fixed left/middle/right zones). |
| **ESC** | Exit application |

After you choose, the tracker starts. Look at a monitor → cursor moves there. Press **q** in the tracker window to quit.

### Other commands

- **Force recalibrate:** `./run.sh --calibrate` — Skips the menu and runs calibration, then starts the tracker.
- **No preview:** `./run.sh --no-preview` — Tracker runs without the camera window.
- **List monitors:** `python face_tracker.py --list-monitors` — Print monitor order (left/middle/right).

## Config

- **`config.json`** — Sensitivity, smoothing, FPS, screen order. See the file for keys.
- **`calibration.json`** — Your face-to-screen mapping (created by calibration). Do not edit by hand.

## Project structure

```
ScreenGaze/
├── face_tracker.py   # Main application
├── config.json       # Configuration
├── requirements.txt  # Python dependencies
├── run.sh            # Launcher
└── README.md
```

`.venv/`, `face_landmarker.task`, and `calibration.json` are local and not committed.

## Notes

- **Wayland:** Cursor movement may require an X11 session.
- **Qt font warnings** on first run are harmless; the app uses OpenCV’s built-in text.
