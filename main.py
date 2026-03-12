from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field

from attractors import AttractorManager, active_attractor_names
from config import (
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    DEFAULT_DT,
    DEFAULT_FOG,
    DEFAULT_LUMINOSITY,
    DEFAULT_PITCH,
    DEFAULT_ROLL,
    DEFAULT_SCALE,
    DEFAULT_SPEED,
    DEFAULT_YAW,
    FIXED_TRAIL_LENGTH,
    FOG_RANGE,
    FOG_STEP_DELTA,
    HAND_TRACKING_FPS,
    MAX_SPEED_POINTS_PER_MINUTE,
    SCALE_RANGE,
    SNAPSHOT_BURN_IN,
    SNAPSHOT_HEIGHT,
    SNAPSHOT_SAMPLE_STRIDE,
    SNAPSHOT_SAMPLES,
    SNAPSHOT_WIDTH,
    SMOOTH_ALPHA,
    SPEED_RANGE,
    PITCH_RANGE,
    YAW_RANGE,
)
from hands import GestureInterpreter

SWITCH_CAPTION_DURATION = 0.85
KEYBOARD_YAW_STEP = 8.0
KEYBOARD_PITCH_STEP = 6.0
KEYBOARD_YAW_HOLD_RATE = 135.0
KEYBOARD_PITCH_HOLD_RATE = 100.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModernGL strange attractor trail viewer")
    parser.add_argument("--no-camera", action="store_true", dest="no_camera", help="Disable webcam hand tracking")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (disables camera)")
    parser.add_argument("--camera-index", type=int, default=-1, dest="camera_index", help="Camera device index to use (-1 = auto-detect built-in camera)")
    parser.add_argument("--headless", action="store_true", help="Run without opening a window; requires snapshot export")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N rendered frames (0 keeps running)")
    parser.add_argument("--screenshot-path", type=str, default="", help="Output path for snapshot-only or headless export")
    parser.add_argument("--snapshot-only", action="store_true", help="Export a Datashader snapshot and exit")
    parser.add_argument("--attractor", type=str, default="", help="Active attractor name for startup or snapshot export")
    parser.add_argument("--snapshot-width", type=int, default=None, help=f"Override snapshot export width (default: {SNAPSHOT_WIDTH})")
    parser.add_argument("--snapshot-height", type=int, default=None, help=f"Override snapshot export height (default: {SNAPSHOT_HEIGHT})")
    parser.add_argument("--snapshot-samples", type=int, default=None, help="Override the default Datashader sample count")
    parser.add_argument("--snapshot-burn-in", type=int, default=None, help="Override the default Datashader burn-in steps")
    parser.add_argument("--snapshot-stride", type=int, default=None, help="Override the default Datashader sample stride")
    return parser.parse_args(argv)


@dataclass
class ControlState:
    yaw: float = DEFAULT_YAW
    pitch: float = DEFAULT_PITCH
    roll: float = DEFAULT_ROLL
    zoom: float = DEFAULT_SCALE
    speed: float = DEFAULT_SPEED
    fog: float = DEFAULT_FOG
    luminosity: float = DEFAULT_LUMINOSITY
    paused: bool = False
    show_overlay: bool = True
    show_shortcuts: bool = True
    show_camera: bool = True
    focus_mode: bool = False
    _targets: dict = field(
        default_factory=lambda: {
            "yaw": DEFAULT_YAW,
            "pitch": DEFAULT_PITCH,
            "zoom": DEFAULT_SCALE,
            "speed": DEFAULT_SPEED,
            "fog": DEFAULT_FOG,
            "luminosity": DEFAULT_LUMINOSITY,
        }
    )

    def set_target(self, name: str, value: float) -> None:
        self._targets[name] = value

    def smooth(self, alpha: float = SMOOTH_ALPHA) -> None:
        for key in ("yaw", "pitch", "zoom", "speed", "fog", "luminosity"):
            current = getattr(self, key)
            target = self._targets[key]
            setattr(self, key, current + alpha * (target - current))
        self.zoom = max(SCALE_RANGE[0], min(SCALE_RANGE[1], self.zoom))
        self.speed = max(SPEED_RANGE[0], min(SPEED_RANGE[1], self.speed))
        self.fog = max(FOG_RANGE[0], min(FOG_RANGE[1], self.fog))
        self.luminosity = max(0.05, min(1.0, self.luminosity))


@dataclass(frozen=True)
class CameraSnapshot:
    frame: object | None
    hand_data: dict


class CameraTrackerSession:
    def __init__(self, capture, tracker, cv2_module) -> None:
        self.capture = capture
        self.tracker = tracker
        self.cv2 = cv2_module
        self._lock = threading.Lock()
        self._running = True
        self._snapshot = CameraSnapshot(frame=None, hand_data={"left": None, "right": None})
        self._worker = threading.Thread(target=self._run, name="camera-tracker", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while self._running:
            try:
                ok, raw_frame = self.capture.read()
            except Exception:
                break

            if not self._running:
                break
            if not ok:
                time.sleep(0.01)
                continue

            camera_frame = self.cv2.flip(raw_frame, 1)
            hand_data = self.tracker.process(camera_frame)
            with self._lock:
                self._snapshot = CameraSnapshot(frame=camera_frame, hand_data=hand_data)

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return self._snapshot

    def close(self) -> None:
        self._running = False
        if self.capture is not None:
            self.capture.release()
        self._worker.join(timeout=1.0)
        self.tracker.close()


def _find_builtin_camera_index() -> int:
    import json
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return 0
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cameras = json.loads(result.stdout).get("SPCameraDataType", [])
        for idx, cam in enumerate(cameras):
            name = cam.get("_name", "").lower()
            model = cam.get("spcamera_model-id", "").lower()
            if "facetime" in name or "facetime" in model:
                return idx
    except Exception:
        pass
    return 0


def maybe_create_camera_session(args: argparse.Namespace) -> CameraTrackerSession | None:
    if args.demo or args.no_camera:
        return None

    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return None

    from hands import HandTracker, MEDIAPIPE_AVAILABLE

    if not MEDIAPIPE_AVAILABLE:
        return None

    cam_index = args.camera_index if args.camera_index >= 0 else _find_builtin_camera_index()
    capture = cv2.VideoCapture(cam_index)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    if hasattr(cv2, "CAP_PROP_FPS"):
        capture.set(cv2.CAP_PROP_FPS, HAND_TRACKING_FPS)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    tracker = None
    try:
        tracker = HandTracker()
        return CameraTrackerSession(capture, tracker, cv2)
    except Exception as exc:
        capture.release()
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        print(f"Camera disabled: {exc}")
        return None


def _adjust_speed(controls: ControlState, delta: float) -> None:
    controls.set_target("speed", max(SPEED_RANGE[0], min(SPEED_RANGE[1], controls._targets["speed"] + delta)))


def _adjust_fog(controls: ControlState, delta: float) -> None:
    controls.set_target("fog", max(FOG_RANGE[0], min(FOG_RANGE[1], controls._targets["fog"] + delta)))


def _adjust_yaw(controls: ControlState, delta: float) -> None:
    controls.set_target("yaw", max(YAW_RANGE[0], min(YAW_RANGE[1], controls._targets["yaw"] + delta)))


def _adjust_pitch(controls: ControlState, delta: float) -> None:
    controls.set_target("pitch", max(PITCH_RANGE[0], min(PITCH_RANGE[1], controls._targets["pitch"] + delta)))


def _apply_held_rotation_controls(controls: ControlState, *, horizontal: float, vertical: float, frame_delta: float) -> None:
    if frame_delta <= 0.0:
        return
    if horizontal:
        _adjust_yaw(controls, horizontal * KEYBOARD_YAW_HOLD_RATE * frame_delta)
    if vertical:
        _adjust_pitch(controls, vertical * KEYBOARD_PITCH_HOLD_RATE * frame_delta)


def _adjust_zoom(controls: ControlState, factor: float) -> None:
    controls.set_target("zoom", max(SCALE_RANGE[0], min(SCALE_RANGE[1], controls._targets["zoom"] * factor)))


def _points_per_second_for_speed(speed: float) -> float:
    clamped_speed = max(SPEED_RANGE[0], min(SPEED_RANGE[1], speed))
    return (MAX_SPEED_POINTS_PER_MINUTE / 60.0) * (clamped_speed / SPEED_RANGE[1])


def _consume_sample_budget(sample_budget: float, speed: float, frame_delta: float) -> tuple[int, float]:
    updated_budget = max(0.0, sample_budget) + (_points_per_second_for_speed(speed) * max(0.0, frame_delta))
    steps = int(updated_budget + 1e-9)
    return steps, updated_budget - steps


def _get_live_render_data(manager: AttractorManager) -> tuple[object, object]:
    return manager.get_render_data(FIXED_TRAIL_LENGTH)


def _apply_slider_control(controls: ControlState, slider_id: str, value: float) -> None:
    if slider_id == "speed":
        controls.set_target("speed", value)
    elif slider_id == "fog":
        controls.set_target("fog", value)
    elif slider_id == "luminosity":
        controls.set_target("luminosity", value)
    elif slider_id == "scale":
        controls.set_target("zoom", value)


def _resolve_snapshot_dimensions(args: argparse.Namespace) -> tuple[int, int]:
    width = SNAPSHOT_WIDTH if args.snapshot_width is None else args.snapshot_width
    height = SNAPSHOT_HEIGHT if args.snapshot_height is None else args.snapshot_height
    if width < 1 or height < 1:
        raise SystemExit("snapshot dimensions must be >= 1")
    return width, height


def run_snapshot_export(args: argparse.Namespace) -> str:
    from renderer import SnapshotRequest, export_attractor_snapshot

    available = active_attractor_names()
    requested_name = args.attractor or available[0]
    lookup = {name.lower(): name for name in available}
    attractor_name = lookup.get(requested_name.lower())
    if attractor_name is None:
        raise SystemExit(f"Unknown active attractor '{requested_name}'. Available: {', '.join(available)}")

    sample_count = SNAPSHOT_SAMPLES if args.snapshot_samples is None else args.snapshot_samples
    burn_in = SNAPSHOT_BURN_IN if args.snapshot_burn_in is None else args.snapshot_burn_in
    sample_stride = SNAPSHOT_SAMPLE_STRIDE if args.snapshot_stride is None else args.snapshot_stride
    width, height = _resolve_snapshot_dimensions(args)
    try:
        request = SnapshotRequest(
            attractor_name=attractor_name,
            yaw=DEFAULT_YAW,
            pitch=DEFAULT_PITCH,
            roll=DEFAULT_ROLL,
            zoom=DEFAULT_SCALE,
            time_value=0.0,
            luminosity=DEFAULT_LUMINOSITY,
            output_path=args.screenshot_path,
            width=width,
            height=height,
            sample_count=sample_count,
            burn_in=burn_in,
            sample_stride=sample_stride,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = export_attractor_snapshot(request)
    print(f"Saved snapshot: {output}")
    return output


def _prime_live_trail(manager: AttractorManager) -> None:
    manager.prime_current_trail(
        dt=DEFAULT_DT,
        sample_count=SNAPSHOT_SAMPLES,
        burn_in=SNAPSHOT_BURN_IN,
        sample_stride=SNAPSHOT_SAMPLE_STRIDE,
    )


def _restart_current_trail(manager: AttractorManager) -> None:
    manager.reset()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    snapshot_width, snapshot_height = _resolve_snapshot_dimensions(args)
    if args.snapshot_only or (args.headless and args.screenshot_path):
        run_snapshot_export(args)
        return
    if args.headless:
        raise SystemExit("--headless now requires --screenshot-path or --snapshot-only")

    import pygame

    from renderer import SceneRenderer, SceneState, SnapshotController, SnapshotRequest

    renderer = SceneRenderer()
    manager = AttractorManager()
    controls = ControlState()
    gestures = GestureInterpreter()
    snapshotter = SnapshotController()

    if args.attractor:
        try:
            manager.switch_to(manager.active_index_for_name(args.attractor))
        except KeyError as exc:
            renderer.quit()
            raise SystemExit(f"Unknown active attractor '{args.attractor}'") from exc
    _prime_live_trail(manager)

    camera_session = maybe_create_camera_session(args)
    camera_frame = None
    running = True
    frame_count = 0
    animation_time = 0.0
    sample_budget = 0.0
    last_frame_time = time.perf_counter()
    active_slider: str | None = None
    hover_slider: str | None = None
    hover_nav_index: int | None = None
    hover_reset = False
    hover_shortcuts_toggle = False
    hover_pip = False
    left_reset_caption_until = 0.0
    right_switch_caption_until = 0.0

    try:
        while running:
            now = time.perf_counter()
            frame_delta = now - last_frame_time
            last_frame_time = now
            if not controls.paused:
                animation_time += frame_delta
            pressed_rotation_keys: set[int] = set()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        controls.paused = not controls.paused
                    elif event.key == pygame.K_r:
                        _restart_current_trail(manager)
                        sample_budget = 0.0
                    elif event.key == pygame.K_s:
                        snapshotter.start(
                            SnapshotRequest(
                                attractor_name=manager.name,
                                yaw=controls.yaw,
                                pitch=controls.pitch,
                                roll=controls.roll,
                                zoom=controls.zoom,
                                time_value=animation_time,
                                luminosity=controls.luminosity,
                                width=snapshot_width,
                                height=snapshot_height,
                                state=manager.state_vector,
                            )
                        )
                    elif event.key == pygame.K_h:
                        controls.show_shortcuts = not controls.show_shortcuts
                    elif event.key == pygame.K_c:
                        controls.show_camera = not controls.show_camera
                    elif event.key == pygame.K_m:
                        controls.focus_mode = not controls.focus_mode
                        controls.show_camera = True
                    elif pygame.K_1 <= event.key < pygame.K_1 + manager.total:
                        manager.switch_to(event.key - pygame.K_1)
                        _prime_live_trail(manager)
                        sample_budget = 0.0
                    elif event.key == pygame.K_UP:
                        _adjust_pitch(controls, -KEYBOARD_PITCH_STEP)
                        pressed_rotation_keys.add(pygame.K_UP)
                    elif event.key == pygame.K_DOWN:
                        _adjust_pitch(controls, KEYBOARD_PITCH_STEP)
                        pressed_rotation_keys.add(pygame.K_DOWN)
                    elif event.key == pygame.K_LEFT:
                        _adjust_yaw(controls, -KEYBOARD_YAW_STEP)
                        pressed_rotation_keys.add(pygame.K_LEFT)
                    elif event.key == pygame.K_RIGHT:
                        _adjust_yaw(controls, KEYBOARD_YAW_STEP)
                        pressed_rotation_keys.add(pygame.K_RIGHT)
                    elif event.unicode == "+" or event.key == pygame.K_KP_PLUS:
                        _adjust_speed(controls, 0.1)
                    elif event.unicode == "-" or event.key == pygame.K_KP_MINUS:
                        _adjust_speed(controls, -0.1)
                    elif event.key == pygame.K_COMMA:
                        _adjust_fog(controls, -FOG_STEP_DELTA)
                    elif event.key == pygame.K_PERIOD:
                        _adjust_fog(controls, FOG_STEP_DELTA)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and controls.show_overlay
                    and not controls.focus_mode
                ):
                    if renderer.shortcuts_toggle_hit_test(manager.names, event.pos, controls.show_shortcuts):
                        controls.show_shortcuts = not controls.show_shortcuts
                        hover_shortcuts_toggle = renderer.shortcuts_toggle_hit_test(manager.names, event.pos, controls.show_shortcuts)
                        active_slider = None
                    else:
                        hovered_nav_index = renderer.navigation_hit_test(manager.names, event.pos)
                        if hovered_nav_index is not None:
                            manager.switch_to(hovered_nav_index)
                            _prime_live_trail(manager)
                            sample_budget = 0.0
                            active_slider = None
                        elif renderer.reset_button_hit_test(manager.names, event.pos):
                            active_slider = None
                            _restart_current_trail(manager)
                            sample_budget = 0.0
                        else:
                            active_slider = renderer.control_hit_test(manager.names, event.pos)
                        hover_nav_index = hovered_nav_index
                    if active_slider is not None:
                        slider_value = renderer.control_value_for_position(manager.names, active_slider, event.pos[0])
                        _apply_slider_control(controls, active_slider, slider_value)
                    hover_slider = active_slider
                    hover_reset = renderer.reset_button_hit_test(manager.names, event.pos)
                    hover_pip = renderer.pip_hit_test(manager.names, event.pos)
                elif event.type == pygame.MOUSEMOTION and controls.show_overlay and not controls.focus_mode:
                    hover_slider = renderer.control_hit_test(manager.names, event.pos)
                    hover_nav_index = renderer.navigation_hit_test(manager.names, event.pos)
                    hover_reset = renderer.reset_button_hit_test(manager.names, event.pos)
                    hover_shortcuts_toggle = renderer.shortcuts_toggle_hit_test(manager.names, event.pos, controls.show_shortcuts)
                    hover_pip = renderer.pip_hit_test(manager.names, event.pos)
                    if active_slider is not None:
                        slider_value = renderer.control_value_for_position(manager.names, active_slider, event.pos[0])
                        _apply_slider_control(controls, active_slider, slider_value)
                        hover_slider = active_slider
                elif event.type == pygame.MOUSEWHEEL:
                    _adjust_zoom(controls, 1.12 if event.y > 0 else 1.0 / 1.12)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        _adjust_zoom(controls, 1.12)
                    elif event.button == 5:
                        _adjust_zoom(controls, 1.0 / 1.12)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    active_slider = None
                elif event.type == pygame.MOUSEMOTION and active_slider is not None:
                    slider_value = renderer.control_value_for_position(manager.names, active_slider, event.pos[0])
                    _apply_slider_control(controls, active_slider, slider_value)
                elif event.type == pygame.WINDOWLEAVE:
                    hover_slider = None
                    hover_nav_index = None
                    hover_reset = False
                    hover_shortcuts_toggle = False
                    hover_pip = False

            hand_data = {"left": None, "right": None}
            camera_frame = None
            if camera_session is not None:
                snapshot = camera_session.snapshot()
                camera_frame = snapshot.frame
                hand_data = snapshot.hand_data

            gesture_frame = gestures.update(hand_data)
            if gesture_frame.yaw is not None:
                controls.set_target("yaw", gesture_frame.yaw)
            if gesture_frame.pitch is not None:
                controls.set_target("pitch", gesture_frame.pitch)
            if gesture_frame.speed is not None:
                controls.set_target("speed", gesture_frame.speed)
            if gesture_frame.fog is not None:
                controls.set_target("fog", gesture_frame.fog)
            if gesture_frame.luminosity is not None:
                controls.set_target("luminosity", gesture_frame.luminosity)
            if gesture_frame.scale is not None:
                controls.set_target("zoom", gesture_frame.scale)

            pressed = pygame.key.get_pressed()
            horizontal = float(pressed[pygame.K_RIGHT] and pygame.K_RIGHT not in pressed_rotation_keys) - float(
                pressed[pygame.K_LEFT] and pygame.K_LEFT not in pressed_rotation_keys
            )
            vertical = float(pressed[pygame.K_DOWN] and pygame.K_DOWN not in pressed_rotation_keys) - float(
                pressed[pygame.K_UP] and pygame.K_UP not in pressed_rotation_keys
            )
            _apply_held_rotation_controls(controls, horizontal=horizontal, vertical=vertical, frame_delta=frame_delta)

            if gesture_frame.reset_current:
                left_reset_caption_until = time.monotonic() + SWITCH_CAPTION_DURATION
                _restart_current_trail(manager)
                sample_budget = 0.0
            if gesture_frame.scene_delta:
                switch_now = time.monotonic()
                right_switch_caption_until = switch_now + SWITCH_CAPTION_DURATION
                manager.switch_relative(gesture_frame.scene_delta)
                _prime_live_trail(manager)
                sample_budget = 0.0

            controls.smooth()

            if not controls.paused:
                steps, sample_budget = _consume_sample_budget(sample_budget, controls.speed, frame_delta)
                if steps > 0:
                    manager.step_many(DEFAULT_DT * controls.speed, steps)
            positions, ages = _get_live_render_data(manager)
            caption_now = time.monotonic()
            left_pip_caption = "Reset" if caption_now < left_reset_caption_until else "Speed"
            right_pip_caption = "Switch" if caption_now < right_switch_caption_until else "Scale"

            renderer.draw(
                SceneState(
                    positions=positions,
                    ages=ages,
                    attractor_name=manager.name,
                    attractor_color=manager.color,
                    attractor_index=manager.index,
                    attractor_total=manager.total,
                    attractor_names=manager.names,
                    attractor_state=manager.state_vector,
                    placard_title=manager.placard.title,
                    placard_year=manager.placard.year,
                    placard_medium=manager.placard.medium,
                    placard_equation=manager.placard.equation,
                    placard_params=manager.placard.params,
                    yaw=controls.yaw,
                    pitch=controls.pitch,
                    roll=controls.roll,
                    zoom=controls.zoom,
                    speed=controls.speed,
                    fog=controls.fog,
                    luminosity=controls.luminosity,
                    point_count=manager.count,
                    fps=renderer.clock.get_fps(),
                    paused=controls.paused,
                    exporting=snapshotter.is_running,
                    export_message=snapshotter.message,
                    show_overlay=controls.show_overlay,
                    show_shortcuts=controls.show_shortcuts,
                    show_camera=controls.show_camera,
                    focus_mode=controls.focus_mode,
                    active_slider=active_slider,
                    hover_slider=hover_slider,
                    hover_nav_index=hover_nav_index,
                    hover_reset=hover_reset,
                    hover_shortcuts_toggle=hover_shortcuts_toggle,
                    hover_pip=hover_pip,
                    left_detected=gesture_frame.left_detected,
                    right_detected=gesture_frame.right_detected,
                    pip_frame=camera_frame,
                    left_landmarks=hand_data["left"],
                    right_landmarks=hand_data["right"],
                    left_pip_caption=left_pip_caption,
                    right_pip_caption=right_pip_caption,
                    time_value=animation_time,
                )
            )

            frame_count += 1
            renderer.tick()

            if args.frames and frame_count >= args.frames:
                running = False
    finally:
        if camera_session is not None:
            camera_session.close()
        snapshotter.close()
        renderer.quit()


if __name__ == "__main__":
    main()
