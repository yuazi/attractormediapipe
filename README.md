# ModernGL Strange Attractor Trail Viewer

Interactive strange attractor viewer built around `pygame`, `moderngl`, `numba`, `datashader`, `opencv-python`, and `mediapipe`. The app renders a single active attractor as a glowing additive point trail in real time, keeps gesture-driven attractor switching and reset, and exports high-resolution density snapshots with Datashader.

![Lorenz UI preview](assets/readme_lorenz_ui.png)

## Features

- Nine active attractors: Lorenz, Aizawa, Sprott B, Thomas, Dadras, Chen, Langford, Rossler, and Halvorsen
- Numba-compiled RK4 stepping and batched sampling for live trails and snapshot exports
- ModernGL real-time renderer with additive point glow and shader-driven pulse / Y-axis drift
- Gesture actions: left pinky touch resets the current attractor, right pinky touch switches to the next
- Grid-based HUD overlay with named study navigation, frosted utility panels, study placard, and live parameter sliders
- Stylized PiP camera panel with optional live webcam feed and MediaPipe hand tracking overlays
- High-resolution Datashader export of the currently active attractor with inferno density coloring

## Install

```bash
python3 -m pip install -r requirements.txt
```

The dependency file is pinned to a tested package set so GitHub installs are reproducible. The HUD ships with bundled `Plus Jakarta Sans`, `Bebas Neue`, and `Onest` font files in `assets/fonts`, so a fresh clone renders with the intended typography. `Neue Haas Grotesk` is treated as an optional local fallback and is not required.

## Run

```bash
python3 main.py
```

Disable the webcam and use keyboard + mouse only:

```bash
python3 main.py --no-camera
```

Export a 5K snapshot without opening the viewer:

```bash
python3 main.py --snapshot-only --attractor Langford --screenshot-path assets/langford_snapshot.png
```

Headless export is routed to the same Datashader path:

```bash
python3 main.py --headless --attractor Chen --screenshot-path assets/chen_snapshot.png
```

Render a larger wallpaper export by overriding the snapshot size:

```bash
python3 main.py --snapshot-only --attractor Langford --snapshot-width 7680 --snapshot-height 4320 --screenshot-path assets/langford_wallpaper.png
```

## Controls

### Webcam gestures

- Left hand thumb + index pinch: simulation speed
- Left hand thumb + ring pinch: luminosity
- Left hand pinky touches palm: reset the current attractor, with `Reset` shown near the pinky-touch area in the PiP
- Right hand palm X: yaw
- Right hand palm Y: pitch
- Right hand thumb + index pinch: zoom
- Right hand thumb + ring pinch: trail length
- Right hand pinky touches palm: switch to the next attractor, with `Switch` shown near the pinky-touch area in the PiP

### Keyboard and mouse

- `ESC`: quit
- `SPACE`: pause / resume
- `R`: restart the current attractor from an empty trail so you can watch it grow again
- `S`: export a 5K Datashader snapshot of the current attractor
- `H`: collapse or expand the shortcuts panel
- `C`: toggle the live camera feed inside the PiP
- `M`: focus mode, showing only the attractor and camera PiP
- `1`-`9`: switch attractors directly
- `LEFT` / `RIGHT`: rotate yaw
- `UP` / `DOWN`: rotate pitch
- `+` / `-`: adjust simulation speed
- `,` / `.`: adjust fog
- Mouse wheel: zoom
- Left mouse drag on overlay sliders: adjust speed, trail length, luminosity, and zoom
- Click `Reset trail` in the parameter panel: restart the current attractor from zero visible history

## UI notes

- The HUD now uses a three-by-three screen grid: coordinates top-left, `(y)us` title top-center, named study navigator top-right, shortcuts mid-left, parameters mid-right, study placard bottom-left, status bottom-center, and the PiP panel bottom-right.
- The PiP panel is always present as part of the HUD chrome. When `C` is enabled and a camera feed is available, the live image renders underneath the scanlines, brackets, and recording/status labels.

## Snapshot details

- The exporter generates `5,000,000` points for the current attractor.
- Default output resolution is `5120x2880`.
- Use `--snapshot-width` and `--snapshot-height` to match your wallpaper resolution or aspect ratio.
- Files are saved as `attractor_YYYYMMDD_HHMMSS.png` unless `--screenshot-path` is provided.
- The exporter sets `NUMBA_CACHE_DIR` automatically so Datashader works on environments where the default cache path is not writable.

## Project layout

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

Run the regression suite:

```bash
python3 -m unittest discover -s tests -v
```
