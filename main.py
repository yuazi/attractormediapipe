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
    DEFAULT_LUMINOSITY,
    DEFAULT_PITCH,
    DEFAULT_ROLL,
    DEFAULT_SCALE,
    DEFAULT_SPEED,
    DEFAULT_TRAIL,
    DEFAULT_YAW,
    HAND_TRACKING_FPS,
    MAX_TRAIL,
    MIN_TRAIL,
    SCALE_RANGE,
    SNAPSHOT_BURN_IN,
    SNAPSHOT_SAMPLE_STRIDE,
    SNAPSHOT_SAMPLES,
    SMOOTH_ALPHA,
    SPEED_RANGE,
    STEPS_PER_FRAME,
    TRAIL_STEP_DELTA,
)
from hands import GestureInterpreter

SWITCH_CAPTION_DURATION = 0.85


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
    parser.add_argument("--snapshot-samples", type=int, default=0, help="Override the default Datashader sample count")
    parser.add_argument("--snapshot-burn-in", type=int, default=0, help="Override the default Datashader burn-in steps")
    parser.add_argument("--snapshot-stride", type=int, default=0, help="Override the default Datashader sample stride")
    return parser.parse_args(argv)


@dataclass
class ControlState:
    yaw: float = DEFAULT_YAW
    pitch: float = DEFAULT_PITCH
    roll: float = DEFAULT_ROLL
    zoom: float = DEFAULT_SCALE
    speed: float = DEFAULT_SPEED
    luminosity: float = DEFAULT_LUMINOSITY
    trail_len: int = DEFAULT_TRAIL
    paused: bool = False
    show_overlay: bool = True
    show_camera: bool = False
    focus_mode: bool = False
    _targets: dict = field(
        default_factory=lambda: {
            "yaw": DEFAULT_YAW,
            "pitch": DEFAULT_PITCH,
            "zoom": DEFAULT_SCALE,
            "speed": DEFAULT_SPEED,
            "luminosity": DEFAULT_LUMINOSITY,
        }
    )

    def set_target(self, name: str, value: float) -> None:
        self._targets[name] = value

    def smooth(self, alpha: float = SMOOTH_ALPHA) -> None:
        for key in ("yaw", "pitch", "zoom", "speed", "luminosity"):
            current = getattr(self, key)
            target = self._targets[key]
            setattr(self, key, current + alpha * (target - current))
        self.zoom = max(SCALE_RANGE[0], min(SCALE_RANGE[1], self.zoom))
        self.speed = max(SPEED_RANGE[0], min(SPEED_RANGE[1], self.speed))
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
    return CameraTrackerSession(capture, HandTracker(), cv2)


def _adjust_speed(controls: ControlState, delta: float) -> None:
    controls.set_target("speed", max(SPEED_RANGE[0], min(SPEED_RANGE[1], controls._targets["speed"] + delta)))


def _adjust_trail(controls: ControlState, delta: int) -> None:
    controls.trail_len = max(MIN_TRAIL, min(MAX_TRAIL, controls.trail_len + delta))


def _adjust_zoom(controls: ControlState, factor: float) -> None:
    controls.set_target("zoom", max(SCALE_RANGE[0], min(SCALE_RANGE[1], controls._targets["zoom"] * factor)))


def _steps_for_speed(speed: float) -> int:
    return max(1, int(round(STEPS_PER_FRAME * max(SPEED_RANGE[0], speed))))


def _apply_slider_control(controls: ControlState, slider_id: str, value: float) -> None:
    if slider_id == "speed":
        controls.set_target("speed", value)
    elif slider_id == "trail_len":
        controls.trail_len = max(MIN_TRAIL, min(MAX_TRAIL, int(round(value))))
    elif slider_id == "luminosity":
        controls.set_target("luminosity", value)
    elif slider_id == "scale":
        controls.set_target("zoom", value)


def run_snapshot_export(args: argparse.Namespace) -> str:
    from renderer import SnapshotRequest, export_attractor_snapshot

    available = active_attractor_names()
    requested_name = args.attractor or available[0]
    lookup = {name.lower(): name for name in available}
    attractor_name = lookup.get(requested_name.lower())
    if attractor_name is None:
        raise SystemExit(f"Unknown active attractor '{requested_name}'. Available: {', '.join(available)}")

    request = SnapshotRequest(
        attractor_name=attractor_name,
        yaw=DEFAULT_YAW,
        pitch=DEFAULT_PITCH,
        roll=DEFAULT_ROLL,
        zoom=DEFAULT_SCALE,
        time_value=0.0,
        luminosity=DEFAULT_LUMINOSITY,
        output_path=args.screenshot_path,
        sample_count=args.snapshot_samples or SNAPSHOT_SAMPLES,
        burn_in=args.snapshot_burn_in or SNAPSHOT_BURN_IN,
        sample_stride=args.snapshot_stride or SNAPSHOT_SAMPLE_STRIDE,
    )
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
    last_frame_time = time.perf_counter()
    active_slider: str | None = None
    left_reset_caption_until = 0.0
    right_switch_caption_until = 0.0

    try:
        while running:
            now = time.perf_counter()
            frame_delta = now - last_frame_time
            last_frame_time = now
            if not controls.paused:
                animation_time += frame_delta

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
                                state=manager.state_vector,
                            )
                        )
                    elif event.key == pygame.K_h:
                        controls.show_overlay = not controls.show_overlay
                    elif event.key == pygame.K_c:
                        controls.show_camera = not controls.show_camera
                    elif event.key == pygame.K_m:
                        controls.focus_mode = not controls.focus_mode
                    elif pygame.K_1 <= event.key < pygame.K_1 + manager.total:
                        manager.switch_to(event.key - pygame.K_1)
                        _prime_live_trail(manager)
                    elif event.key == pygame.K_UP:
                        _adjust_speed(controls, 0.1)
                    elif event.key == pygame.K_DOWN:
                        _adjust_speed(controls, -0.1)
                    elif event.key == pygame.K_LEFT:
                        _adjust_trail(controls, -TRAIL_STEP_DELTA)
                    elif event.key == pygame.K_RIGHT:
                        _adjust_trail(controls, TRAIL_STEP_DELTA)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and controls.show_overlay
                    and not controls.focus_mode
                ):
                    if renderer.reset_button_hit_test(manager.names, event.pos):
                        active_slider = None
                        _restart_current_trail(manager)
                    else:
                        active_slider = renderer.control_hit_test(manager.names, event.pos)
                    if active_slider is not None:
                        slider_value = renderer.control_value_for_position(manager.names, active_slider, event.pos[0])
                        _apply_slider_control(controls, active_slider, slider_value)
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
            if gesture_frame.luminosity is not None:
                controls.set_target("luminosity", gesture_frame.luminosity)
            if gesture_frame.scale is not None:
                controls.set_target("zoom", gesture_frame.scale)
            if gesture_frame.trail_len is not None:
                controls.trail_len = max(MIN_TRAIL, min(MAX_TRAIL, int(round(gesture_frame.trail_len))))
            if gesture_frame.reset_current:
                left_reset_caption_until = time.monotonic() + SWITCH_CAPTION_DURATION
                _restart_current_trail(manager)
            if gesture_frame.scene_delta:
                switch_now = time.monotonic()
                right_switch_caption_until = switch_now + SWITCH_CAPTION_DURATION
                manager.switch_relative(gesture_frame.scene_delta)
                _prime_live_trail(manager)

            controls.smooth()

            if not controls.paused:
                manager.step_many(DEFAULT_DT * controls.speed, _steps_for_speed(controls.speed))
            positions, ages = manager.get_render_data(controls.trail_len)
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
                    luminosity=controls.luminosity,
                    trail_len=controls.trail_len,
                    point_count=manager.count,
                    fps=renderer.clock.get_fps(),
                    paused=controls.paused,
                    exporting=snapshotter.is_running,
                    export_message=snapshotter.message,
                    show_overlay=controls.show_overlay,
                    show_camera=controls.show_camera,
                    focus_mode=controls.focus_mode,
                    active_slider=active_slider,
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
        renderer.quit()


if __name__ == "__main__":
    main()
