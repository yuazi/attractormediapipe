from __future__ import annotations

from screenshot.presets import available_snapshot_presets

WIN_W = 1800
WIN_H = 1100
FPS = 60

DEFAULT_DT = 0.005
MAX_SPEED_POINTS_PER_MINUTE = 500_000
CONTROL_SMOOTHING_FRAMES = 12
DEFAULT_POINT_SIZE = 2.8
TRAIL_DECAY = 0.9

PARTICLE_SIZE_MAX = 7
PARTICLE_SIZE_MIN = 1
PARTICLE_ALPHA_MAX = 128
PARTICLE_ALPHA_MIN = 8
GLOW_SPRITE_SIZE = 48

LIVE_SAMPLE_COUNT = 5_000_000
TRAIL_BUFFER_CAPACITY = LIVE_SAMPLE_COUNT
FIXED_TRAIL_LENGTH = 100_000

PIP_W = 400
PIP_H = 225

CAMERA_FRAME_WIDTH = 960
CAMERA_FRAME_HEIGHT = 540
CAMERA_CAPTURE_FPS = 60
TRACKING_FRAME_WIDTH = 640
TRACKING_FRAME_HEIGHT = 360
HAND_TRACKING_FPS = 24

BACKGROUND_COLOR = (0, 0, 0)
SKELETON_COLOR = (255, 255, 255)

CAMERA_DISTANCE = 5.0

DEFAULT_LUMINOSITY = 0.82
DEFAULT_FOG = 1.0
DEFAULT_SCALE = 1.6
DEFAULT_SPEED = 1.0
DEFAULT_YAW = 0.0
DEFAULT_PITCH = 0.0
DEFAULT_ROLL = 0.0

YAW_RANGE = (-180.0, 180.0)
PITCH_RANGE = (-90.0, 90.0)
SCALE_RANGE = (0.5, 3.5)
SPEED_RANGE = (0.1, 3.5)
LUMINOSITY_RANGE = (0.05, 1.0)
FOG_RANGE = (0.0, 1.0)
FOG_STEP_DELTA = 0.05

PINCH_RANGE = (0.02, 0.25)

SCENE_TURN_COOLDOWN_SECONDS = 0.55
SCENE_SWITCH_PINKY_TOUCH_MAX_DISTANCE = 0.11

BLOOM_DOWNSAMPLE = 4
BLOOM_BLUR_PASSES = 1
BLOOM_ALPHA = 24

CAPTION = "(y)us particle attractor"
SCREENSHOT_PREFIX = "attractor"
SCREENSHOT_DIR = "screenshot"
SNAPSHOT_LOG_PATH = "assets/snapshot_log.jsonl"
SNAPSHOT_WIDTH = 5120
SNAPSHOT_HEIGHT = 2880
SNAPSHOT_SAMPLES = LIVE_SAMPLE_COUNT
SNAPSHOT_BURN_IN = 25_000
SNAPSHOT_SAMPLE_STRIDE = 1
DEFAULT_SNAPSHOT_PRESET = "nebula"
SNAPSHOT_PRESET_NAMES = available_snapshot_presets()
PRESET_HUD_FLASH_SECONDS = 2.0

HELP_LINES = [
    "LEFT HAND",
    "Thumb + index pinch -> Speed",
    "Thumb + ring pinch -> Luminosity",
    "Pinky touch palm -> Reset attractor",
    "",
    "RIGHT HAND",
    "Palm X -> Yaw",
    "Palm Y -> Pitch",
    "Thumb + index pinch -> Scale / zoom",
    "Thumb + ring pinch -> Fog",
    "Pinky touch palm -> Switch attractor",
    "",
    "KEYS",
    "[1-9] switch  [R] restart trail  [SPACE] pause/resume",
    "[H] shortcuts  [C] camera  [M] pip mode",
    "[S] snapshot  [P] preset  [G] ghost",
    "[ESC] quit  [ARROWS] rotate",
    "[+/-] speed  [,/.] fog  [WHEEL] zoom",
]
