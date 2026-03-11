from __future__ import annotations

from dataclasses import dataclass


WIN_W = 1800
WIN_H = 1100
FPS = 60

DEFAULT_DT = 0.005
STEPS_PER_FRAME = 4
SMOOTH_ALPHA = 0.08
DEFAULT_POINT_SIZE = 2.8

PARTICLE_SIZE_MAX = 7
PARTICLE_SIZE_MIN = 1
PARTICLE_ALPHA_MAX = 128
PARTICLE_ALPHA_MIN = 8
GLOW_SPRITE_SIZE = 48

MIN_TRAIL = 20_000
LIVE_SAMPLE_COUNT = 5_000_000
MAX_TRAIL = 100_000
DEFAULT_TRAIL = 20_000
TRAIL_STEP_DELTA = 1000
TRAIL_BUFFER_CAPACITY = LIVE_SAMPLE_COUNT
PARTICLE_COUNT_RANGE = (1, 10)
DEFAULT_PARTICLE_COUNT = 5

PIP_W = 320
PIP_H = 180
PIP_MARGIN = 12
OVERLAY_W = 700
OVERLAY_H = 260

CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
HAND_TRACKING_FPS = 60

BACKGROUND_COLOR = (0, 0, 0)
HUD_PANEL_COLOR = (8, 10, 18, 190)
HUD_PANEL_BORDER = (84, 68, 44)
HUD_TEXT = (226, 205, 163)
HUD_MUTED = (122, 102, 72)
HUD_BAR_BG = (26, 21, 14)
HUD_BAR_FILL = (190, 151, 82)
HUD_HELP_TEXT = (194, 173, 138)
STATUS_OK = (219, 188, 122)
STATUS_LOST = (96, 78, 56)
PIP_BORDER_COLOR = (128, 102, 68, 110)
SKELETON_COLOR = (255, 255, 255)

CAMERA_DISTANCE = 5.0
CAMERA_FOV = 520.0

DEFAULT_LUMINOSITY = 0.82
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

PINCH_RANGE = (0.02, 0.25)

SCENE_TURN_COOLDOWN_SECONDS = 0.55
SCENE_SWITCH_PINKY_TOUCH_MAX_DISTANCE = 0.11

BLOOM_DOWNSAMPLE = 4
BLOOM_BLUR_PASSES = 1
BLOOM_ALPHA = 24

CAPTION = "(y)us particle attractor"
SCREENSHOT_PREFIX = "attractor"
SNAPSHOT_WIDTH = 3840
SNAPSHOT_HEIGHT = 2160
SNAPSHOT_SAMPLES = LIVE_SAMPLE_COUNT
SNAPSHOT_BURN_IN = 25_000
SNAPSHOT_SAMPLE_STRIDE = 1

HELP_LINES = [
    "LEFT HAND",
    "Thumb + index pinch -> Speed",
    "Thumb + ring pinch -> Luminosity",
    "Pinky touch palm -> Previous attractor",
    "",
    "RIGHT HAND",
    "Palm X -> Yaw",
    "Palm Y -> Pitch",
    "Thumb + index pinch -> Scale / zoom",
    "Thumb + ring pinch -> Trail length",
    "Pinky touch palm -> Next attractor",
    "",
    "KEYS",
    "[1-7] switch  [R] restart trail  [SPACE] pause",
    "[H] overlay  [C] camera  [M] pip mode",
    "[S] snapshot",
    "[ESC] quit  [UP/DOWN] speed",
    "[LEFT/RIGHT] trail length  [WHEEL] zoom",
]


@dataclass(frozen=True)
class AttractorSpec:
    name: str
    color: tuple[int, int, int]
    scale_hint: float


ATTRACTOR_SPECS = (
    AttractorSpec("Lorenz", (255, 88, 88), 1.0),
    AttractorSpec("Aizawa", (255, 177, 74), 2.0),
    AttractorSpec("Sprott B", (226, 112, 255), 1.2),
    AttractorSpec("Thomas", (98, 255, 229), 1.5),
    AttractorSpec("Dadras", (89, 164, 255), 1.0),
    AttractorSpec("Chen", (77, 115, 255), 1.0),
    AttractorSpec("Langford", (255, 86, 171), 2.0),
)
