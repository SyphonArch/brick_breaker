import numpy as np

# Dimensions of a brick
WIDTH = 120
HEIGHT = 75

DIM_X = 6
DIM_Y = 9

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

BALL_COLOR = (0, 100, 255)
TRAIL = False
TRAIL_COLOR = (0, 255, 255)

RADIUS = 10  # radius of balls
BORDER = 2  # border width of bricks
DOUBLE_HIT_THRESHOLD = 4

SPEED = 10  # speed of balls
SPACING = 40  # spacing between balls, approximate

ANGLE_MIN = 10

ARROW_HEAD_RADIUS = 10
ARROW_THICKNESS = 4
ARROW_MAX_LENGTH = 200

FPS = 144

EARLY_TERMINATION = True

EXPLORER_RESOLUTION = 128  # how many angles to subdivide into for exploration

# Don't touch
assert (RADIUS + SPEED) * 2 < HEIGHT
assert (RADIUS + SPEED) * 2 < WIDTH
RES_X = WIDTH * DIM_X
RES_Y = HEIGHT * DIM_Y

SPEED_LIMIT = min(x // 2 - RADIUS for x in (WIDTH, HEIGHT))
assert SPEED <= SPEED_LIMIT

R2 = RADIUS ** 2

RELPOS_XY = np.array([WIDTH, HEIGHT])
RELPOS_X = np.array([WIDTH, 0])
RELPOS_Y = np.array([0, HEIGHT])

FLIP_X = np.array([-1, 1])
FLIP_Y = np.array([1, -1])
FLIP_XY = np.array([-1, -1])

MIRROR_XY = np.array([RADIUS * 2, RADIUS * 2])
MIRROR_X = np.array([RADIUS * 2, 0])
MIRROR_Y = np.array([0, RADIUS * 2])

ANGLE_MAX = 180 - ANGLE_MIN

# 180 is added to the angles as rotation is clockwise! (mind the inverted y-axis)
ANGLE_MIN_RAD = (180 + ANGLE_MIN) / 180 * np.pi
ANGLE_MAX_RAD = (180 + ANGLE_MAX) / 180 * np.pi
