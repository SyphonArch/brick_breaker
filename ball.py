from constants import *
import pygame
import numpy as np

R2 = RADIUS ** 2

X_MIN = RADIUS + SPEED
X_MAX = WIDTH - X_MIN

Y_MIN = RADIUS + SPEED
Y_MAX = HEIGHT - Y_MIN

RELPOS_XY = np.array([WIDTH, HEIGHT])
RELPOS_X = np.array([WIDTH, 0])
RELPOS_Y = np.array([0, HEIGHT])

FLIP_X = np.array([-1, 1])
FLIP_Y = np.array([1, -1])
FLIP_XY = np.array([-1, -1])

MIRROR_XY = np.array([RADIUS * 2, RADIUS * 2])
MIRROR_X = np.array([RADIUS * 2, 0])
MIRROR_Y = np.array([0, RADIUS * 2])


def safe_access_grid(grid, seg_x, seg_y, decrement=False):
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


def get_walls(grid, seg_x, seg_y, x_flip, y_flip):
    dx = 1 if x_flip else -1
    dy = 1 if y_flip else -1

    left = safe_access_grid(grid, seg_x + dx, seg_y)
    diag = safe_access_grid(grid, seg_x + dx, seg_y + dy)
    top = safe_access_grid(grid, seg_x, seg_y + dy)

    return left, diag, top


def decrement_bricks(grid, seg_x, seg_y, x_flip, y_flip, dec_left, dec_diag, dec_top):
    dx = 1 if x_flip else -1
    dy = 1 if y_flip else -1
    safe_access_grid(grid, seg_x + dx, seg_y, dec_left)
    safe_access_grid(grid, seg_x + dx, seg_y + dy, dec_diag)
    safe_access_grid(grid, seg_x, seg_y + dy, dec_top)


def get_rel_values(position, vector, x_flip, y_flip):
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


def in_seg(seg_start, seg_end, point):
    x, y = point
    min_x, max_x = sorted([seg_start[0], seg_end[0]])
    min_y, max_y = sorted([seg_start[1], seg_end[1]])
    return min_x < x <= max_x and min_y < y <= max_y


def reflect(position, flip_x, flip_y):
    if flip_x and flip_y:
        return FLIP_XY - position
    elif flip_x:
        return np.array([RADIUS * 2 - position[0], position[1]])
    elif flip_y:
        return np.array([position[0], RADIUS * 2 - position[1]])
    else:
        return position


def get_circle_intersections(start, vector):
    assert any(vector)
    if all(vector):
        dx, dy = vector
        a = dy / dx
        b = start[1] - a * start[0]
        # y = ax + b
        # x ^ 2 + y ^ 2 = RADIUS ^ 2
        A = a ** 2 + 1
        B = 2 * a * b
        C = b ** 2 - R2
        D = B ** 2 - 4 * A * C
        if D <= 0:
            return []
        else:
            x1 = (- B - D ** 0.5) / (2 * A)
            x2 = (- B + D ** 0.5) / (2 * A)
            y1 = a * x1 + b
            y2 = a * x2 + b
    elif vector[0]:
        y1 = start[1]
        if y1 < RADIUS:
            y2 = y1
            x1 = (R2 - y1 ** 2) ** 0.5
            x2 = -x1
        else:
            return []
    else:
        x1 = start[0]
        if x1 < RADIUS:
            x2 = x1
            y1 = (R2 - x1 ** 2) ** 0.5
            y2 = -y1
        else:
            return []
    return [np.array([x1, y1]), np.array([x2, y2])]


def get_rotation(vec1, vec2):
    x1, y1 = vec1
    x2, y2 = vec2
    return np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)


def rotate(vector, angle):
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rot_mat @ vector


def dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def length(vector):
    return np.linalg.norm(vector)


def normalize(vector, size):
    return vector * size / length(vector)


def circle_reflect(start, hit_point, vector):
    traveled = dist(start, hit_point)
    vec_len = length(vector)
    to_travel = vec_len - traveled

    half_rotation = get_rotation(vector, - hit_point)
    full_rotation = half_rotation * 2
    new_vector = - rotate(vector, full_rotation)
    new_pos = hit_point + new_vector * to_travel / vec_len
    return new_pos, new_vector


class Ball:
    def __init__(self, screen, position, velocity, grid, points):
        self.screen = screen
        self.position = np.asarray(position, float)
        self.velocity = np.asarray(velocity, float)
        self.grid = grid
        self.points = points
        self.collected_points = 0

    def draw(self):
        if TRAIL:
            rad = np.pi / 180
            trirad = RADIUS - 1
            rotation = np.arctan2(*(-self.velocity))
            pygame.draw.polygon(self.screen, TRAIL_COLOR,
                                ((self.position[0] + trirad * np.sin(rotation) * 4,
                                  self.position[1] + trirad * np.cos(rotation) * 4),
                                 (self.position[0] + trirad * np.sin(rotation - 120 * rad),
                                  self.position[1] + trirad * np.cos(rotation - 120 * rad)),
                                 (self.position[0] + trirad * np.sin(rotation + 120 * rad),
                                  self.position[1] + trirad * np.cos(rotation + 120 * rad))))
        pygame.draw.circle(self.screen, BALL_COLOR, tuple(self.position), RADIUS)

    def tick(self):
        self.collect_points()
        if self.needs_collision_checking():
            self.collision_safe_move()
        else:
            self.position += self.velocity
        # Check for early termination.
        if EARLY_TERMINATION:
            seg_x = int(self.position[0] // WIDTH)
            seg_y = int(self.position[1] // HEIGHT)
            if self.velocity[1] > 0 and \
                all(not any(line) for line in self.grid[seg_y:]) and \
                    all(not any(line) for line in self.points[seg_y:]) and \
                        safe_access_grid(self.grid, seg_x - 1, seg_y - 1) <= 0 and \
                            safe_access_grid(self.grid, seg_x + 1, seg_y - 1) <= 0:
                self.position[1] = HEIGHT * DIM_Y + SPEED * 10

    def collect_points(self):
        seg_x, rem_x = divmod(self.position[0], WIDTH)
        seg_y, rem_y = divmod(self.position[1], HEIGHT)
        seg_x = int(seg_x)
        seg_y = int(seg_y)
        if safe_access_grid(self.points, seg_x, seg_y) == 1:
            if abs(rem_x - WIDTH // 2) < RADIUS * 2 and abs(rem_y - HEIGHT // 2) < RADIUS * 2:
                self.collected_points += 1
                self.points[seg_y][seg_x] = 0

    def needs_collision_checking(self):
        rem_x = self.position[0] % WIDTH
        rem_y = self.position[1] % HEIGHT
        if X_MIN < rem_x < X_MAX and Y_MIN < rem_y < Y_MAX:
            return False
        else:
            return True

    def collision_safe_move(self):
        if not any(self.velocity):
            self.position += self.velocity
            return

        seg_x, rem_x = divmod(self.position[0], WIDTH)
        seg_y, rem_y = divmod(self.position[1], HEIGHT)
        seg_x = int(seg_x)
        seg_y = int(seg_y)

        x_flip = rem_x > (WIDTH // 2)
        y_flip = rem_y > (HEIGHT // 2)

        left, diag, top = get_walls(self.grid, seg_x, seg_y, x_flip, y_flip)

        if not left and not diag and not top:
            self.position += self.velocity
            return

        offset_x = seg_x * WIDTH
        offset_y = seg_y * HEIGHT
        offset = np.array([offset_x, offset_y])

        rem_pos = self.position - offset

        rel_pos, rel_vec = get_rel_values(rem_pos, self.velocity, x_flip, y_flip)
        rel_end = rel_pos + rel_vec

        dec_left, dec_diag, dec_top = False, False, False

        if left and top:  # 2 cases
            if rel_end[0] <= RADIUS:
                dec_left = True
                rel_end = MIRROR_X + rel_end * FLIP_X
                rel_vec = rel_vec * FLIP_X
            if rel_end[1] <= RADIUS:
                dec_top = True
                rel_end = MIRROR_Y + rel_end * FLIP_Y
                rel_vec = rel_vec * FLIP_Y
        elif left and diag:  # 1 case
            if rel_end[0] <= RADIUS:
                ratio_a = RADIUS - rel_end[0]
                ratio_b = rel_pos[0] - RADIUS
                intersection_y = (rel_pos[1] * ratio_a + rel_end[1] * ratio_b) / (ratio_a + ratio_b)
                if intersection_y > - DOUBLE_HIT_THRESHOLD:
                    dec_left = True
                if intersection_y < DOUBLE_HIT_THRESHOLD:
                    dec_diag = True
                rel_end = MIRROR_X + rel_end * FLIP_X
                rel_vec = rel_vec * FLIP_X
        elif diag and top:  # 1 case
            if rel_end[1] <= RADIUS:
                ratio_a = RADIUS - rel_end[1]
                ratio_b = rel_pos[1] - RADIUS
                intersection_x = (rel_pos[0] * ratio_a + rel_end[0] * ratio_b) / (ratio_a + ratio_b)
                if intersection_x > - DOUBLE_HIT_THRESHOLD:
                    dec_top = True
                if intersection_x < DOUBLE_HIT_THRESHOLD:
                    dec_diag = True
                rel_end = MIRROR_Y + rel_end * FLIP_Y
                rel_vec = rel_vec * FLIP_Y
        elif left:  # 1 case
            if rel_end[0] <= RADIUS:
                if rel_vec[0]:
                    y = (RADIUS - rel_pos[0]) / rel_vec[0] * rel_vec[1] + rel_pos[1]
                    if y > 0:
                        dec_left = True
                        rel_end = MIRROR_X + rel_end * FLIP_X
                        rel_vec = rel_vec * FLIP_X
                    else:
                        if rel_vec[1]:
                            intersections = get_circle_intersections(rel_pos, rel_vec)
                            if intersections:
                                intersections.sort(key=lambda point: point[0])
                                if in_seg(rel_pos, rel_end, intersections[1]):
                                    dec_left = True
                                    hit = intersections[1]
                                    rel_end, rel_vec = circle_reflect(rel_pos, hit, rel_vec)

        elif top:  # 1 case
            if rel_end[1] <= RADIUS:
                if rel_vec[1]:
                    x = (RADIUS - rel_pos[1]) / rel_vec[1] * rel_vec[0] + rel_pos[0]
                    if x > 0:
                        dec_top = True
                        rel_end = MIRROR_Y + rel_end * FLIP_Y
                        rel_vec = rel_vec * FLIP_Y
                    else:
                        if rel_vec[0]:
                            intersections = get_circle_intersections(rel_pos, rel_vec)
                            if intersections:
                                intersections.sort(key=lambda point: point[1])
                                if in_seg(rel_pos, rel_end, intersections[1]):
                                    dec_top = True
                                    hit = intersections[1]
                                    rel_end, rel_vec = circle_reflect(rel_pos, hit, rel_vec)
        elif diag:  # 1 case
            intersections = get_circle_intersections(rel_pos, rel_vec)
            if intersections:
                intersections.sort(key=lambda point: dist(point, rel_pos))
                candidate = intersections[0]
                if in_seg(rel_pos, rel_end, candidate):
                    if candidate[0] > 0 and candidate[1] > 0:
                        dec_diag = True
                        hit = candidate
                        rel_end, rel_vec = circle_reflect(rel_pos, hit, rel_vec)
            if not dec_diag:
                if rel_end[0] <= RADIUS and rel_end[1] <= 0:
                    dec_diag = True
                    rel_end = MIRROR_X + rel_end * FLIP_X
                    rel_vec = rel_vec * FLIP_X
                if rel_end[1] <= RADIUS and rel_end[0] <= 0:
                    dec_diag = True
                    rel_end = MIRROR_Y + rel_end * FLIP_Y
                    rel_vec = rel_vec * FLIP_Y
        else:  # 1 case - but should have been dealt with earlier
            raise AssertionError("Won't be seeing this.")

        decrement_bricks(self.grid, seg_x, seg_y, x_flip, y_flip, dec_left, dec_diag, dec_top)
        real_end, real_vec = get_rel_values(rel_end, rel_vec, x_flip, y_flip)

        # To prevent infinite loops
        if abs(real_vec[1]) < 0.1:
            real_vec = rotate(real_vec, 0.09)

        self.position = real_end + offset
        self.velocity = real_vec
