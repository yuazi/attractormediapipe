from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass, field

from attractors import AttractorManager
from config import (
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    DEFAULT_DT,
    DEFAULT_LUMINOSITY,
    DEFAULT_PARTICLE_COUNT,
    DEFAULT_PITCH,
    DEFAULT_ROLL,
    DEFAULT_SCALE,
    DEFAULT_SPEED,
    DEFAULT_TRAIL,
    DEFAULT_YAW,
    HAND_TRACKING_FPS,
    MAX_TRAIL,
    MIN_TRAIL,
    PARTICLE_COUNT_RANGE,
    SMOOTH_ALPHA,
    SPEED_RANGE,
    STEPS_PER_FRAME,
    WIN_H,
    WIN_W,
)
from hands import GestureInterpreter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand-controlled strange attractor visualizer")
    parser.add_argument("--no-camera", action="store_true", dest="no_camera", help="Disable webcam hand tracking")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (disables camera)")
    parser.add_argument("--camera-index", type=int, default=-1, dest="camera_index", help="Camera device index to use (-1 = auto-detect built-in camera)")
    parser.add_argument("--headless", action="store_true", help="Use SDL dummy video driver")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N rendered frames (0 keeps running)")
    parser.add_argument("--screenshot-path", type=str, default="", help="Save the final rendered frame to a specific path")
    return parser.parse_args(argv)


@dataclass
class ControlState:
    yaw: float = DEFAULT_YAW
    pitch: float = DEFAULT_PITCH
    roll: float = DEFAULT_ROLL
    scale: float = DEFAULT_SCALE
    speed: float = DEFAULT_SPEED
    luminosity: float = DEFAULT_LUMINOSITY
    trail_len: int = DEFAULT_TRAIL
    particle_count: int = DEFAULT_PARTICLE_COUNT
    bloom_enabled: bool = True
    show_overlay: bool = False
    show_camera: bool = False
    focus_mode: bool = False
    _targets: dict = field(default_factory=lambda: {
        "yaw": DEFAULT_YAW,
        "pitch": DEFAULT_PITCH,
        "roll": DEFAULT_ROLL,
        "scale": DEFAULT_SCALE,
        "speed": DEFAULT_SPEED,
        "luminosity": DEFAULT_LUMINOSITY,
        "trail_len": float(DEFAULT_TRAIL),
    })

    def set_target(self, name: str, value: float) -> None:
        self._targets[name] = value

    def smooth(self, alpha: float = SMOOTH_ALPHA) -> None:
        for key in ("yaw", "pitch", "roll", "scale", "speed", "luminosity"):
            current = getattr(self, key)
            target = self._targets[key]
            setattr(self, key, current + alpha * (target - current))
        self.trail_len = int(round(self._targets["trail_len"]))


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


def _apply_particle_count(controls: ControlState, raw_value: str) -> str:
    if not raw_value:
        return str(controls.trail_len)
    particle_count = max(MIN_TRAIL, min(MAX_TRAIL, int(raw_value)))
    controls.trail_len = particle_count
    controls.set_target("trail_len", float(particle_count))
    return str(particle_count)


def _apply_particle_stream_count(controls: ControlState, raw_value: float | int) -> None:
    minimum, maximum = PARTICLE_COUNT_RANGE
    controls.particle_count = max(minimum, min(maximum, int(round(raw_value))))


def _apply_slider_control(controls: ControlState, slider_id: str, raw_value: float) -> None:
    if slider_id == "speed":
        controls.speed = raw_value
        controls.set_target("speed", raw_value)
    elif slider_id == "trail_len":
        particle_count = int(round(raw_value))
        controls.trail_len = particle_count
        controls.set_target("trail_len", float(particle_count))
    elif slider_id == "particle_count":
        _apply_particle_stream_count(controls, raw_value)
    elif slider_id == "luminosity":
        controls.luminosity = raw_value
        controls.set_target("luminosity", raw_value)
    elif slider_id == "scale":
        controls.scale = raw_value
        controls.set_target("scale", raw_value)


def demo_rotation_targets(frame_count: int) -> tuple[float, float, float]:
    t = frame_count / 60.0
    yaw = math.sin(t * 0.85) * 160.0
    pitch = math.sin(t * 0.41) * 36.0
    roll = math.cos(t * 0.33) * 24.0
    return yaw, pitch, roll


def _find_builtin_camera_index() -> int:
    """On macOS, try to find the built-in FaceTime camera index via system_profiler.
    Falls back to 0 on failure or non-macOS systems."""
    import platform
    import subprocess
    import json
    if platform.system() != "Darwin":
        return 0
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    from renderer import SceneRenderer, SceneState

    renderer = SceneRenderer()
    manager = AttractorManager()
    controls = ControlState()
    gestures = GestureInterpreter()

    camera_session = maybe_create_camera_session(args)
    camera_frame = None
    frame_count = 0
    particle_input_active = False
    particle_input_text = str(DEFAULT_TRAIL)
    active_slider: str | None = None
    running = True

    try:
        while running:
            fps = renderer.clock.get_fps()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if particle_input_active:
                        if event.key == pygame.K_ESCAPE:
                            particle_input_active = False
                            particle_input_text = str(controls.trail_len)
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            particle_input_text = _apply_particle_count(controls, particle_input_text)
                            particle_input_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            particle_input_text = particle_input_text[:-1]
                        elif event.unicode.isdigit() and len(particle_input_text) < 5:
                            particle_input_text += event.unicode
                        continue

                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        manager.reset()
                    elif event.key == pygame.K_s:
                        screenshot = renderer.save_screenshot()
                        print(f"Saved screenshot: {screenshot}")
                    elif event.key == pygame.K_b:
                        controls.bloom_enabled = not controls.bloom_enabled
                    elif event.key == pygame.K_h:
                        controls.show_overlay = not controls.show_overlay
                    elif event.key == pygame.K_c:
                        controls.show_camera = not controls.show_camera
                    elif event.key == pygame.K_m:
                        controls.focus_mode = not controls.focus_mode
                    elif event.key == pygame.K_p:
                        particle_input_active = True
                        particle_input_text = ""
                    elif pygame.K_1 <= event.key < pygame.K_1 + manager.total:
                        manager.switch_to(event.key - pygame.K_1)
                    elif event.key == pygame.K_UP:
                        controls.set_target("speed", min(SPEED_RANGE[1], controls._targets["speed"] + 0.1))
                    elif event.key == pygame.K_DOWN:
                        controls.set_target("speed", max(SPEED_RANGE[0], controls._targets["speed"] - 0.1))
                    elif event.key == pygame.K_RIGHT:
                        particle_input_text = _apply_particle_count(controls, str(int(controls._targets["trail_len"]) + 200))
                    elif event.key == pygame.K_LEFT:
                        particle_input_text = _apply_particle_count(controls, str(int(controls._targets["trail_len"]) - 200))
                    elif event.key == pygame.K_PERIOD:
                        _apply_particle_stream_count(controls, controls.particle_count + 1)
                    elif event.key == pygame.K_COMMA:
                        _apply_particle_stream_count(controls, controls.particle_count - 1)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not controls.focus_mode:
                    active_slider = renderer.hud.control_hit_test(event.pos)
                    if active_slider is not None:
                        _apply_slider_control(controls, active_slider, renderer.hud.control_value_for_position(active_slider, event.pos[0]))
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    active_slider = None
                elif event.type == pygame.MOUSEMOTION and active_slider is not None and not controls.focus_mode:
                    _apply_slider_control(controls, active_slider, renderer.hud.control_value_for_position(active_slider, event.pos[0]))

            hand_data = {"left": None, "right": None}
            camera_frame = None
            if camera_session is not None:
                snapshot = camera_session.snapshot()
                camera_frame = snapshot.frame
                hand_data = snapshot.hand_data

            gesture_frame = gestures.update(hand_data)

            if gesture_frame.left_detected:
                if gesture_frame.yaw is not None:
                    controls.set_target("yaw", gesture_frame.yaw)
                if gesture_frame.pitch is not None:
                    controls.set_target("pitch", gesture_frame.pitch)
                controls.set_target("roll", DEFAULT_ROLL)
            else:
                demo_yaw, demo_pitch, demo_roll = demo_rotation_targets(frame_count)
                controls.set_target("yaw", demo_yaw)
                controls.set_target("pitch", demo_pitch)
                controls.set_target("roll", demo_roll)

            if gesture_frame.right_detected:
                if active_slider != "luminosity" and gesture_frame.luminosity is not None:
                    controls.set_target("luminosity", gesture_frame.luminosity)
                if active_slider != "scale" and gesture_frame.scale is not None:
                    controls.set_target("scale", gesture_frame.scale)

            if active_slider != "speed" and gesture_frame.speed is not None:
                controls.set_target("speed", gesture_frame.speed)

            if gesture_frame.scene_delta:
                manager.switch_relative(gesture_frame.scene_delta)

            controls.smooth()

            dt = DEFAULT_DT * controls.speed
            manager.step_many(dt, STEPS_PER_FRAME)
            points_2d, depths = manager.get_projected_trail(
                controls.trail_len,
                controls.yaw,
                controls.pitch,
                controls.roll,
                controls.scale,
                (WIN_W, WIN_H),
            )

            renderer.draw(
                SceneState(
                    points_2d=points_2d,
                    depths=depths,
                    attractor_name=manager.name,
                    attractor_color=manager.color,
                    placard_title=manager.placard.title,
                    placard_year=manager.placard.year,
                    placard_medium=manager.placard.medium,
                    placard_params=manager.placard.params,
                    luminosity=controls.luminosity,
                    speed=controls.speed,
                    scale=controls.scale,
                    trail_len=controls.trail_len,
                    particle_count=controls.particle_count,
                    attractor_names=manager.names,
                    attractor_index=manager.index,
                    attractor_state=manager.state_vector,
                    point_count=len(points_2d),
                    fps=fps,
                    left_detected=gesture_frame.left_detected,
                    right_detected=gesture_frame.right_detected,
                    bloom_enabled=controls.bloom_enabled,
                    show_overlay=controls.show_overlay,
                    show_camera=controls.show_camera,
                    focus_mode=controls.focus_mode,
                    active_slider=active_slider,
                    particle_input_active=particle_input_active,
                    particle_input_text=particle_input_text,
                    pip_frame=camera_frame,
                    left_landmarks=hand_data["left"],
                    right_landmarks=hand_data["right"],
                )
            )

            frame_count += 1
            renderer.tick()

            if args.frames and frame_count >= args.frames:
                running = False
    finally:
        if args.screenshot_path:
            renderer.save_screenshot(args.screenshot_path)
        if camera_session is not None:
            camera_session.close()
        renderer.quit()


if __name__ == "__main__":
    main()
