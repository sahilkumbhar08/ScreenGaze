# ScreenGaze (Compatible with Linux Distro)

Move your cursor to the screen you're looking at. Uses your webcam and head pose to switch between monitors—no need to drag the mouse.

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

On first run the app will download the face model once (~3 MB).

## Run

```bash
./run.sh
```

- Look at a monitor → cursor moves there. Press **q** to quit.
- No preview window: `./run.sh --no-preview`
- List monitors: `python face_tracker.py --list-monitors`

## Config

Edit `config.json` to adjust sensitivity, smoothing, and screen order.
