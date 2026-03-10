# Hand-Controlled Strange Attractor Visualizer

Interactive strange attractor renderer built with `pygame`, `numpy`, `opencv-python`, and `mediapipe`. The app renders each attractor as a glowing particle cloud with an optional bloom pass, exposes hand-driven camera and visual controls, and overlays a white hand skeleton PiP feed in the bottom-left corner when a webcam is available.

![Demo screenshot](assets/attractor_screenshot.png)

## Features

- Seven 3D attractors: Lorenz, Rossler, Halvorsen, Thomas, Dadras, Aizawa, and Sprott B
- RK4 integration with bounded ring-buffer trails and perspective projection
- Cached glow sprites with age-based size/alpha gradients
- `pygame.surfarray` bloom pass for soft additive halos
- Left-hand yaw/pitch controls with pinch-based speed control
- Adjustable particle streams (`1..10`) plus a separate trail-length control
- Left-hand pinky-to-palm touch for previous attractor switching
- Right-hand pinky-to-palm touch for next attractor switching
- Demo fallback when the webcam or tracking dependencies are unavailable
- Live PiP webcam preview with hand skeleton overlay
- Headless fixed-frame mode for automated screenshots

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

If you want to disable the webcam (demo/auto-rotate mode):

```bash
python3 main.py --no-camera
```

If you want to force demo mode:

```bash
python3 main.py --demo
```

If you want to render a fixed number of demo frames and save a screenshot:

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg python3 main.py --demo --headless --frames 600 --screenshot-path assets/attractor_screenshot.png
```

## Gesture Cheat Sheet

| Hand | Gesture | Effect |
| --- | --- | --- |
| Left | Palm X | Yaw rotation |
| Left | Palm Y | Pitch rotation |
| Left | Pinch | Simulation speed |
| Left | Pinky touches palm | Previous attractor |
| Right | Index fingertip Y | Luminosity |
| Right | Thumb-index pinch | Scale / zoom |
| Right | Pinky touches palm | Next attractor |

## Keyboard Controls

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `R` | Reset current attractor and clear the trail |
| `S` | Save a timestamped screenshot |
| `B` | Toggle bloom |
| `H` | Toggle the operator overlay |
| `C` | Toggle the camera PiP |
| `M` | Toggle focus mode (camera, attractor, and footer only) |
| `1`-`7` | Switch attractors directly |
| `UP` / `DOWN` | Adjust simulation speed |
| `LEFT` / `RIGHT` | Adjust trail length by `200` |
| `,` / `.` | Adjust particle stream count (`1..10`) |
| `P` | Type an exact trail length in the HUD |

## Project Layout

```text
.
├── attractors/
├── assets/
├── hands/
├── renderer/
├── tests/
├── config.py
├── main.py
└── requirements.txt
```

## Testing

Run the pure-function regression suite with:

```bash
python3 -m unittest discover -s tests -v
```

The implementation was also smoke-tested locally with:

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg python3 main.py --demo --headless --frames 600 --screenshot-path assets/attractor_screenshot.png
```
