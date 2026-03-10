from __future__ import annotations

from dataclasses import dataclass


WIN_W = 1800
WIN_H = 1100
FPS = 60

DEFAULT_DT = 0.005
STEPS_PER_FRAME = 3
SMOOTH_ALPHA = 0.08

PARTICLE_SIZE_MAX = 7
PARTICLE_SIZE_MIN = 1
PARTICLE_ALPHA_MAX = 128
PARTICLE_ALPHA_MIN = 8
GLOW_SPRITE_SIZE = 48

MIN_TRAIL = 500
MAX_TRAIL = 8000
DEFAULT_TRAIL = 3200
TRAIL_BUFFER_CAPACITY = MAX_TRAIL + 512
PARTICLE_COUNT_RANGE = (1, 10)
DEFAULT_PARTICLE_COUNT = 3

PIP_W = 320
PIP_H = 180
PIP_MARGIN = 12

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

CAMERA_DISTANCE = 4.6
CAMERA_FOV = 520.0

DEFAULT_LUMINOSITY = 0.68
DEFAULT_SCALE = 1.6
DEFAULT_SPEED = 1.0
DEFAULT_YAW = 0.0
DEFAULT_PITCH = 0.0
DEFAULT_ROLL = 0.0

YAW_RANGE = (-180.0, 180.0)
PITCH_RANGE = (-90.0, 90.0)
SCALE_RANGE = (0.3, 3.5)
SPEED_RANGE = (0.1, 3.5)
LUMINOSITY_RANGE = (0.05, 1.0)

PINCH_RANGE = (0.02, 0.25)
INDEX_Y_RANGE = (0.10, 0.90)

SCENE_TURN_COOLDOWN_SECONDS = 0.55
SCENE_SWITCH_PINKY_TOUCH_MAX_DISTANCE = 0.11

BLOOM_DOWNSAMPLE = 4
BLOOM_BLUR_PASSES = 3
BLOOM_ALPHA = 140

CAPTION = "(y)us particle attractor"
SCREENSHOT_PREFIX = "attractor"

HELP_LINES = [
    "LEFT HAND",
    "Palm X -> Yaw",
    "Palm Y -> Pitch",
    "Pinch -> Speed",
    "Pinky touch palm -> Previous attractor",
    "",
    "RIGHT HAND",
    "Index Y -> Luminosity",
    "Pinch -> Scale / zoom",
    "Pinky touch palm -> Next attractor",
    "",
    "KEYS",
    "[1-8] switch  [R] reset  [B] bloom",
    "[H] overlay  [C] camera  [S] screenshot",
    "[M] focus mode",
    "[ESC] quit",
    "[UP/DOWN] speed  [LEFT/RIGHT] trail length",
    "[,/.] particles  [P] type trail length",
]


@dataclass(frozen=True)
class AttractorSpec:
    name: str
    color: tuple[int, int, int]
    scale_hint: float


ATTRACTOR_SPECS = (
    AttractorSpec("Lorenz", (0, 255, 136), 1.0),
    AttractorSpec("Rossler", (0, 207, 255), 1.2),
    AttractorSpec("Halvorsen", (255, 215, 0), 0.8),
    AttractorSpec("Thomas", (255, 255, 255), 1.5),
    AttractorSpec("Dadras", (170, 68, 255), 1.0),
    AttractorSpec("Aizawa", (224, 232, 255), 2.0),
    AttractorSpec("Sprott B", (255, 34, 85), 1.2),
    AttractorSpec("Chen", (255, 122, 90), 1.0),
)
