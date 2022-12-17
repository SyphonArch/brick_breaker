import numpy.typing as npt
from constants import *
import physics

from random import shuffle, randint, choice
import pygame


def safe_access_grid(grid: list[list[int]], seg_x: int, seg_y: int, decrement: bool = False) -> int:
    """Safely access given segment of given grid.

    Returns the value at the segment if valid segment is specified.

    Returns -1 if the specified segment is out of bounds, and part of the top, left or right wall.
    Returns 0 if the specified segment is out of bounds, and part of the baseline."""
    if 0 <= seg_x < DIM_X and 0 <= seg_y < DIM_Y:
        if decrement:
            assert grid[seg_y][seg_x] > 0
            grid[seg_y][seg_x] -= 1
        return grid[seg_y][seg_x]
    else:
        if seg_y < DIM_Y:
            return -1
        else:
            return 0


def get_surroundings(grid: list[list[int]], seg_x: int, seg_y: int, x_flip: bool, y_flip: bool) -> tuple[int, int, int]:
    """Check the left, top, and diagonal segments and return grid values.

    X and Y axes may be flipped."""
    dx = 1 if x_flip else -1
    dy = 1 if y_flip else -1

    left = safe_access_grid(grid, seg_x + dx, seg_y)
    diag = safe_access_grid(grid, seg_x + dx, seg_y + dy)
    top = safe_access_grid(grid, seg_x, seg_y + dy)

    return left, diag, top


def decrement_bricks(grid: list[list[int]], seg_x: int, seg_y: int, x_flip: bool, y_flip: bool,
                     dec_left: bool, dec_diag: bool, dec_top: bool) -> None:
    """Decrement the left, top, and diagonal segments by 1 or 0."""
    dx = 1 if x_flip else -1
    dy = 1 if y_flip else -1
    safe_access_grid(grid, seg_x + dx, seg_y, dec_left)
    safe_access_grid(grid, seg_x + dx, seg_y + dy, dec_diag)
    safe_access_grid(grid, seg_x, seg_y + dy, dec_top)


def get_rel_values(position: npt.NDArray[float], vector: npt.NDArray[float], x_flip: bool, y_flip: bool) \
        -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Flip the position and velocity of ball within its segment."""
    if x_flip:
        if y_flip:
            rel_pos = RELPOS_XY + position * FLIP_XY
            rel_vec = vector * FLIP_XY
        else:
            rel_pos = RELPOS_X + position * FLIP_X
            rel_vec = vector * FLIP_X
    else:
        if y_flip:
            rel_pos = RELPOS_Y + position * FLIP_Y
            rel_vec = vector * FLIP_Y
        else:
            rel_pos = position
            rel_vec = vector
    return rel_pos, rel_vec


def brick_color(value: int) -> npt.NDArray[int]:
    """Given the value of a brick, return its RGB value."""
    _full_red = 30
    _full_blue = 40
    assert _full_blue > _full_red
    redness = min(value / _full_red, 1)
    blueness = min((value - _full_red) / _full_blue, 1)
    return np.clip(np.array([230 - blueness * 200, 230 - redness * 230, blueness * 70]).astype(int), 0, 255)


def gridpos_to_coordinates(i: int, j: int) -> tuple[int, int]:
    """Given the coordinate of a grid position, return the pixel coordinates."""
    x, y = j * WIDTH, i * HEIGHT
    return x, y


def draw_bricks(screen: pygame.Surface, grid: list[list[int]], font: pygame.font.Font) -> None:
    """Draw the bricks onto the screen."""
    for i in range(len(grid)):
        line = grid[i]
        for j in range(len(line)):
            value = line[j]
            if value:
                x, y = gridpos_to_coordinates(i, j)
                pygame.draw.rect(screen, brick_color(value), (x, y, WIDTH, HEIGHT))
                pygame.draw.rect(screen, WHITE, (x, y, WIDTH, HEIGHT), width=BORDER)
                value_text = font.render(str(value), True, WHITE)
                text_rect = value_text.get_rect(center=(x + WIDTH // 2, y + HEIGHT // 2))
                screen.blit(value_text, text_rect)


def draw_points(screen: pygame.Surface, points: list[list[int]]) -> None:
    """Draw the points onto the screen."""
    for i in range(len(points)):
        line = points[i]
        for j in range(len(line)):
            if line[j]:
                x, y = gridpos_to_coordinates(i, j)
                pygame.draw.circle(screen, GREEN, (x + WIDTH // 2, y + HEIGHT // 2), RADIUS)


def rand_gen(grid: list[list[int]], points: list[list[int]], n: int) -> None:
    """Generate the bricks and points, given parameter n."""
    max_bricks = min(n // 10 + 2, DIM_X - 1)
    min_bricks = max(max_bricks - 3, 1)
    brick_count = randint(min_bricks, max_bricks)

    assert all(grid[0][i] == 0 for i in range(DIM_X))
    assert all(points[0][i] == 0 for i in range(DIM_X))

    placement = [n] * brick_count + [0] * (DIM_X - brick_count)
    shuffle(placement)
    grid[0] = placement
    empties = [i for i in range(len(placement)) if placement[i] == 0]
    point_idx = choice(empties)
    points[0][point_idx] = 1


def shift_down(grid: list[list[int]], points: list[list[int]]) -> tuple[bool, int]:
    """Shift down the grid.

    Returns a tuple, where the first value is whether a brick has reached the floor,
    and the second value is the number of points that have reached the floor.
    """
    assert all(grid[DIM_Y - 1][i] == 0 for i in range(DIM_X))
    for i in range(DIM_Y - 1, 0, -1):
        grid[i] = grid[i - 1]
    grid[0] = [0] * DIM_X

    for i in range(DIM_Y - 1, 0, -1):
        points[i] = points[i - 1]
    points[0] = [0] * DIM_X

    taken_points = sum(points[DIM_Y - 1])
    points[DIM_Y - 1] = [0] * DIM_X

    return any(grid[DIM_Y - 1]), taken_points


def draw_arrow(screen: pygame.Surface, color: tuple[int, int, int],
               start: npt.NDArray[float], end: npt.NDArray[float],
               trirad: int = 10, thickness: int = 4) -> None:
    """Draw an arrow from given start to given end."""
    lcolor = color
    tricolor = color
    rad = np.pi / 180
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = np.arctan2(start[1] - end[1], end[0] - start[0]) + np.pi / 2
    pygame.draw.polygon(screen, tricolor,
                        ((end[0] + trirad * np.sin(rotation),
                          end[1] + trirad * np.cos(rotation)),
                         (end[0] + trirad * np.sin(rotation - 120 * rad),
                          end[1] + trirad * np.cos(rotation - 120 * rad)),
                         (end[0] + trirad * np.sin(rotation + 120 * rad),
                          end[1] + trirad * np.cos(rotation + 120 * rad))))


def draw_arrow_modified(screen: pygame.Surface, color: tuple[int, int, int],
                        start: npt.NDArray[float], vector: npt.NDArray[float], length: int) -> None:
    """Draw an arrow starting from start, clipped by length, in vector direction."""
    if not any(vector):
        return
    vector *= min(length / physics.length(vector), 1)
    end = start + vector
    draw_arrow(screen, color, start, end)


def clipped_direction(vector: npt.NDArray[float]) -> npt.NDArray[float]:
    """Given a vector, clip the angle it makes with the x-axis."""
    angle = np.arctan2(vector[1], vector[0]) + np.pi * 2
    target_angle = min(MAX_ANGLE_RAD, max(MIN_ANGLE_RAD, angle))
    return physics.rotate(vector, target_angle - angle)
