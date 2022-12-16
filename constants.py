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

BALL_COLOR = (0, 100, 255)
TRAIL = False
TRAIL_COLOR = (0, 255, 255)

RADIUS = 10  # radius of balls
BORDER = 2  # border width of bricks
DOUBLE_HIT_THRESHOLD = 4

SPEED = 10  # speed of balls

INTERVAL = 40 // SPEED  # interval between balls in frames

MIN_ANGLE = 10

FPS = 144

EARLY_TERMINATION = True

# Don't touch
assert (RADIUS + SPEED) * 2 < HEIGHT
assert (RADIUS + SPEED) * 2 < WIDTH
RES_X = WIDTH * DIM_X
RES_Y = HEIGHT * DIM_Y
