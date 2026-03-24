# attractormediapipe

Interactive strange-attractor performance tool built with `pygame`, `moderngl`, `numba`, `datashader`, `opencv-python`, and `mediapipe`.

The app renders one attractor at a time as a glowing particle trail, lets you steer the scene with hand gestures or traditional controls, and exports high-resolution wallpaper snapshots with preset-based styling and embedded metadata.

<video src="assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>

## What It Does

At runtime, `attractormediapipe` combines three systems:

1. A live simulation loop that advances one of nine chaotic systems in real time.
2. A ModernGL renderer that draws the newest portion of the trail as additive point sprites over a procedural atmosphere.
3. A camera + gesture layer that maps hand motion to camera rotation, speed, luminosity, fog, zoom, attractor switching, and reset/randomize actions.

When you press `S`, or run in `--snapshot-only` mode, the app switches to a Datashader export pipeline that generates millions of points, rasterizes them at wallpaper resolution, writes both clean and textured PNG variants, embeds descriptive PNG text metadata, and appends a JSONL archive entry to `assets/snapshot_log.jsonl`.

## Active Attractors

The viewer currently ships with nine attractors:

- Lorenz
- Aizawa
- Sprott B
- Thomas
- Dadras
- Chen
- Langford
- Rossler
- Halvorsen

Each attractor exposes:

- A Numba-compiled RK4 stepper for CPU fallback and snapshot generation.
- A parameter dictionary for HUD display, preset metadata, and controlled randomization.
- A GLSL-compatible parameter layout so the live renderer can use a GPU transform-feedback stepper when the active OpenGL context supports it.

## Rendering Model

The live renderer is built around a few layered ideas rather than a traditional mesh scene:

- The trail is stored as a ring buffer of 3D points.
- The newest `100,000` points are normalized and drawn as additive point sprites.
- Age-based fading is applied in the fragment shader instead of by mutating per-point alpha on the CPU.
- A procedural background texture and separate fog veil keep the scene readable without flattening the attractor.
- Attractor switches cut directly to the newly selected system.
- When available, the live simulation can advance through a transform-feedback GPU stepper; if that path fails or is unsupported, the code falls back to the existing Numba CPU stepping path automatically.

This means the app remains usable on machines with modest graphics support, but can offload more work to the GPU when the context allows it.

## Snapshot Pipeline

Snapshot export is intentionally different from the live render.

- It generates `5,000,000` points by default.
- It defaults to `5120x2880`.
- It uses Datashader for dense 2D accumulation rather than simply saving the live framebuffer.
- It produces two files for every export:
  - `*_clean.png`
  - `*_textured.png`

The clean image is the direct preset-based density render. The textured image applies a preset-specific surface treatment:

| Preset | Palette | Texture treatment | Background | Mood |
| --- | --- | --- | --- | --- |
| `nebula` | warm fire-like gradient | grain + scanlines | near-black | cinematic / warm |
| `blueprint` | monochrome blue range | none | deep navy | technical |
| `void` | viridis-inspired range | vignette | pure black | minimal |
| `print` | greyscale | halftone | white | printable |

Each PNG includes these text metadata fields:

- `Attractor`
- `Preset`
- `Parameters`
- `Points`
- `Resolution`
- `Timestamp`
- `Generator`

Each export also appends a JSON object with the same information plus file paths to `assets/snapshot_log.jsonl`.

## Main Features

- Real-time rendering of nine strange attractors.
- MediaPipe-based left/right hand interpretation.
- Gesture-driven pinky touch actions for reset and attractor switching.
- Smooth interpolation for speed, luminosity, fog, and zoom changes.
- Ghost mode for freezing the trail while still orbiting the camera.
- Focus mode with a slim performance bar for stage-friendly operation.
- Snapshot preset cycling in the live viewer.
- Headless and snapshot-only export support.
- GPU trail stepping path with CPU fallback.
- Unittest regression coverage for controls, snapshots, renderer behavior, and GPU/CPU equivalence helpers.

## Installation

Install the pinned dependency set:

```bash
python3 -m pip install -r requirements.txt
```

### Dependencies in plain language

- `pygame`: window creation, input handling, and the app loop.
- `moderngl`: OpenGL context and shader management.
- `numba`: JIT acceleration for attractor stepping and sampling.
- `opencv-python`: camera capture.
- `mediapipe`: hand landmark detection.
- `datashader` + `pandas`: high-resolution snapshot rasterization.
- `Pillow`: HUD composition, image output, and PNG metadata writing.

The project includes bundled fonts in `assets/fonts`, so a fresh checkout should render the intended UI typography without requiring system font setup.

## Running The App

Start the interactive viewer:

```bash
python3 main.py
```

Run without camera input:

```bash
python3 main.py --no-camera
```

Run in demo mode, which also disables the camera path:

```bash
python3 main.py --demo
```

Open on a specific camera index:

```bash
python3 main.py --camera-index 1
```

Render only a fixed number of frames, which is useful for smoke checks:

```bash
python3 main.py --frames 300
```

## Snapshot And Headless Usage

Export a snapshot and exit:

```bash
python3 main.py --snapshot-only --attractor Lorenz --preset blueprint --screenshot-path screenshot/lorenz_blueprint.png
```

Export with a larger wallpaper resolution:

```bash
python3 main.py --snapshot-only --attractor Lorenz --snapshot-width 7680 --snapshot-height 4320 --preset nebula --snapshot-fit contain --screenshot-path screenshot/lorenz_8k.png
```

Headless export uses the same snapshot renderer:

```bash
python3 main.py --headless --attractor Chen --preset void --screenshot-path screenshot/chen_void.png
```

### Important snapshot notes

- `--headless` currently requires `--snapshot-only` or `--screenshot-path`.
- Invalid preset names exit cleanly and print the valid preset list.
- `--snapshot-fit cover` fills the frame and may crop; `--snapshot-fit contain` keeps the full attractor visible with margin.
- Snapshot filenames default to the `screenshot/` directory with a timestamped base name.
- If `--screenshot-path` is provided, the code preserves the `_clean` / `_textured` naming convention automatically.

## CLI Reference

| Flag | Meaning |
| --- | --- |
| `--no-camera` | Disable webcam hand tracking. |
| `--demo` | Force a no-camera demo run. |
| `--camera-index` | Choose a specific webcam; `-1` scans a small range automatically. |
| `--headless` | Run without opening a window. |
| `--frames` | Exit after a fixed number of rendered frames. |
| `--screenshot-path` | Base output path for snapshot export. |
| `--snapshot-only` | Export once and exit. |
| `--attractor` | Select the startup or export attractor by name. |
| `--snapshot-width` | Override snapshot width. |
| `--snapshot-height` | Override snapshot height. |
| `--snapshot-samples` | Override Datashader point count. |
| `--snapshot-burn-in` | Override burn-in steps before sampling. |
| `--snapshot-stride` | Override sample stride. |
| `--snapshot-fit` | Choose `cover` to fill the frame or `contain` to keep the full attractor visible. |
| `--preset` | Select one of `nebula`, `blueprint`, `void`, or `print`. |

## Controls

### Webcam gestures

The app treats the left and right hands differently.

| Gesture | Result |
| --- | --- |
| Left thumb + index pinch | Change simulation speed |
| Left thumb + ring pinch | Change luminosity |
| Left pinky touch on palm | Reset the current trail |
| Right palm X | Yaw |
| Right palm Y | Pitch |
| Right thumb + index pinch | Zoom / scale |
| Right thumb + ring pinch | Fog |
| Right pinky touch on palm | Next attractor |

### Keyboard and mouse

| Input | Result |
| --- | --- |
| `ESC` | Quit |
| `SPACE` | Pause / resume, or exit ghost mode |
| `R` | Restart the current trail |
| `S` | Export a snapshot using the active preset |
| `P` | Cycle snapshot presets |
| `G` | Toggle ghost mode |
| `H` | Show / hide the shortcuts panel |
| `C` | Show / hide the camera feed |
| `M` | Toggle focus mode |
| `1` to `9` | Switch directly to an attractor |
| `LEFT` / `RIGHT` | Adjust yaw |
| `UP` / `DOWN` | Adjust pitch |
| `+` / `-` | Adjust speed |
| `,` / `.` | Adjust fog |
| Mouse wheel | Zoom |
| Drag slider tracks | Adjust speed, fog, luminosity, or scale from the HUD |

## Performance-Oriented Interaction Notes

These behaviors matter if you want to use the app as a live visual instrument rather than a static demo:

- Parameter changes are smoothed over `12` frames instead of snapping immediately.
- Ghost mode freezes trail growth but keeps camera movement live, so you can orbit a fixed form.
- Focus mode removes most chrome and shows a slim bottom bar with attractor name, preset, fps, and `GHOST` status.
- Attractor changes switch immediately so performance gestures feel direct.
- Preset changes update the status area immediately, even though the live renderer is not a literal Datashader preview.

## UI Layout

The default HUD is organized into a deliberate studio-style layout:

- Top-left: current attractor position.
- Top-center: title lockup.
- Top-right: attractor navigator.
- Mid-left: shortcuts panel.
- Mid-right: parameter controls.
- Bottom-left: study placard.
- Bottom-center: fps and state summary.
- Bottom-right: picture-in-picture camera panel.

The PiP camera panel stays part of the interface even when the live camera feed is hidden. When a camera is active, it shows a stylized preview with gesture overlays and short status captions near the active control regions.

## Project Structure

```text
.
├── attractors/         # one file per attractor + shared manager/base classes
├── assets/             # fonts, model files, screenshots, export log
├── hands/              # tracker, gesture logic, skeleton drawing
├── renderer/           # scene renderer, GPU stepper, snapshot export helpers
├── screenshot/         # snapshot preset definitions
├── tests/              # unittest regression suite
├── config.py           # global tunables and ranges
├── main.py             # application entry point and event loop
└── requirements.txt    # pinned dependency set
```

## Development Notes

### Why there are both CPU and GPU stepping paths

The CPU path is still important:

- It is the stable fallback when a system cannot support the transform-feedback route.
- It powers snapshot generation.
- It keeps tests deterministic and makes numerical equivalence checks easier.

The GPU path exists to reduce per-frame Python overhead during steady-state live rendering.

### Why snapshots are not just screenshots

The live renderer is optimized for interactivity. The snapshot renderer is optimized for dense, printable output. That is why the exported images:

- regenerate the attractor from simulation data,
- use millions of points,
- apply preset-based offline styling,
- and embed metadata for later cataloging.

## Testing

Run the regression suite with:

```bash
python3 -m unittest discover -s tests -v
```

The test suite covers:

- attractor stability,
- gesture timing and mappings,
- snapshot output and metadata,
- renderer state wiring,
- and GPU/CPU stepper equivalence helpers.

Some GPU-specific tests may skip automatically on machines that cannot create a standalone OpenGL context.

## Troubleshooting

### No camera available

Use:

```bash
python3 main.py --no-camera
```

The rest of the app still works with keyboard and mouse.

### Snapshot preset error

If `--preset` uses an unknown name, rerun with one of:

- `nebula`
- `blueprint`
- `void`
- `print`

### Slow first snapshot import

Datashader and Matplotlib-related initialization can be slower on the first run. The exporter sets `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` automatically to improve compatibility on systems where the default cache directories are not writable.
