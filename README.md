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

### Glasses Reminder Feature

ScreenGaze automatically detects if you're wearing computer glasses and reminds you every 2 minutes (configurable) to protect your eyes.

**How it works:**
- The app uses computer vision to detect glasses frames in the eye region
- Edge detection identifies glasses frames (horizontal lines across eyes)
- When glasses are detected: No reminders are sent
- When no glasses detected for 2 minutes: You get a desktop notification reminder
- A "👓 ON/OFF" indicator and countdown timer appear in the preview window

**Controls:**
- **q** - Quit the application

**Configuration:**
```json
{
  "enable_glasses_reminder": true,
  "glasses_reminder_interval": 120
}
```

- `enable_glasses_reminder`: Set to `false` to disable
- `glasses_reminder_interval`: Time in seconds between reminders (default: 120 = 2 minutes)

**Note:** Detection works best with well-lit faces and clear glasses frames. The system analyzes the eye region for frame-like structures and horizontal edges characteristic of glasses.
