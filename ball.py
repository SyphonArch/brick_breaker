from constants import *
import logic
import physics

import pygame
import numpy as np
import numpy.typing as npt


class Ball:
    def __init__(self, screen: pygame.Surface, position: npt.NDArray[float], velocity: npt.NDArray[float],
                 grid: list[list[int]], points: list[list[int]], frame_offset: int | float, speed: int):
        """Initializes Ball.

        Note that frame_offset should be set to the number of frames that have passed since the first ball of the round
        has been launched."""
        self.screen = screen
        self.position = np.asarray(position, float)
        self.velocity = np.asarray(velocity, float)
        self.grid = grid
        self.points = points
        self.collected_points = 0
        self.frame_offset = frame_offset
        self.terminated = False

        self.X_MIN = RADIUS + speed
        self.X_MAX = WIDTH - self.X_MIN
        self.Y_MIN = RADIUS + speed
        self.Y_MAX = HEIGHT - self.Y_MIN

    def draw(self) -> None:
        """Draw itself onto predetermined screen."""
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

    def tick(self) -> None:
        """Move the ball by one frame, and check for termination."""
        if self.terminated:
            print("Attempted to move a terminated ball!")
            return

        if EARLY_TERMINATION:
            self.early_termination_check()

        self.collect_points()

        # To prevent infinite loops
        if abs(self.velocity[1]) < 0.1:
            if self.velocity[1] * self.velocity[0] >= 0:
                self.velocity = physics.rotate(self.velocity, 0.01)
            else:
                self.velocity = physics.rotate(self.velocity, -0.01)

        if self.needs_collision_checking():
            self.collision_safe_move()
        else:
            self.position += self.velocity
        self.frame_offset += 1

        if self.position[1] >= RES_Y:
            self.frame_offset -= 1
            self.position -= self.velocity
            self.terminate()

    def early_termination_check(self) -> None:
        """Check if the ball can be terminated early, and do so if conditions are met."""
        seg_x = int(self.position[0] // WIDTH)
        seg_y = int(self.position[1] // HEIGHT)
        if self.velocity[1] > 0 and all(not any(line) for line in self.grid[seg_y:]) and \
                all(not any(line) for line in self.points[seg_y:]) and \
                logic.safe_access_grid(self.grid, seg_x - 1, seg_y - 1) <= 0 and \
                logic.safe_access_grid(self.grid, seg_x + 1, seg_y - 1) <= 0:
            self.terminate()

    def terminate(self) -> None:
        """Given a ball about to hit the baseline, calculate where on the x-axis the ball will hit,
        and terminate."""
        vector_multiplier = (RES_Y - self.position[1]) / self.velocity[1]
        self.position += vector_multiplier * self.velocity
        boundary_crosses, rem = divmod(self.position[0], RES_X)
        if boundary_crosses % 2:
            rem = RES_X - rem
        self.position[0] = rem
        self.position[1] = 0
        self.frame_offset += vector_multiplier
        self.terminated = True

    def collect_points(self) -> None:
        """Check whether points can be collected - if so, increment self.collected_points."""
        seg_x, rem_x = divmod(self.position[0], WIDTH)
        seg_y, rem_y = divmod(self.position[1], HEIGHT)
        seg_x = int(seg_x)
        seg_y = int(seg_y)
        if logic.safe_access_grid(self.points, seg_x, seg_y) == 1:
            if abs(rem_x - WIDTH // 2) < RADIUS * 2 and abs(rem_y - HEIGHT // 2) < RADIUS * 2:
                self.collected_points += 1
                self.points[seg_y][seg_x] = 0

    def needs_collision_checking(self) -> bool:
        """Determine whether the ball is in potential proximity of a wall or brick."""
        rem_x = self.position[0] % WIDTH
        rem_y = self.position[1] % HEIGHT
        if self.X_MIN < rem_x < self.X_MAX and self.Y_MIN < rem_y < self.Y_MAX:
            return False
        else:
            return True

    def collision_safe_move(self) -> None:
        """Move the ball, resolving any collisions that happen."""
        if not any(self.velocity):
            self.position += self.velocity
            return

        seg_x, rem_x = divmod(self.position[0], WIDTH)
        seg_y, rem_y = divmod(self.position[1], HEIGHT)
        seg_x = int(seg_x)
        seg_y = int(seg_y)

        x_flip = rem_x > (WIDTH // 2)
        y_flip = rem_y > (HEIGHT // 2)

        left, diag, top = logic.get_surroundings(self.grid, seg_x, seg_y, x_flip, y_flip)

        if not left and not diag and not top:  # 1 case
            self.position += self.velocity
            return

        offset_x = seg_x * WIDTH
        offset_y = seg_y * HEIGHT
        offset = np.array([offset_x, offset_y])

        rem_pos = self.position - offset

        rel_pos, rel_vec = logic.get_rel_values(rem_pos, self.velocity, x_flip, y_flip)
        rel_end = rel_pos + rel_vec

        dec_left, dec_diag, dec_top = False, False, False

        if left and top:  # 2 cases (with diag and without)
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
                            intersections = Ball.get_circle_intersections(rel_pos, rel_vec)
                            if intersections:
                                intersections.sort(key=lambda point: point[0])
                                if physics.in_seg(rel_pos, rel_end, intersections[1]):
                                    dec_left = True
                                    hit = intersections[1]
                                    rel_end, rel_vec = Ball.circle_reflect(rel_pos, hit, rel_vec)

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
                            intersections = Ball.get_circle_intersections(rel_pos, rel_vec)
                            if intersections:
                                intersections.sort(key=lambda point: point[1])
                                if physics.in_seg(rel_pos, rel_end, intersections[1]):
                                    dec_top = True
                                    hit = intersections[1]
                                    rel_end, rel_vec = Ball.circle_reflect(rel_pos, hit, rel_vec)
        elif diag:  # 1 case
            intersections = Ball.get_circle_intersections(rel_pos, rel_vec)
            if intersections:
                intersections.sort(key=lambda point: physics.dist(point, rel_pos))
                candidate = intersections[0]
                if physics.in_seg(rel_pos, rel_end, candidate):
                    if candidate[0] > 0 and candidate[1] > 0:
                        dec_diag = True
                        hit = candidate
                        rel_end, rel_vec = Ball.circle_reflect(rel_pos, hit, rel_vec)
            if not dec_diag:
                if rel_end[0] <= RADIUS and rel_end[1] <= 0:
                    dec_diag = True
                    rel_end = MIRROR_X + rel_end * FLIP_X
                    rel_vec = rel_vec * FLIP_X
                if rel_end[1] <= RADIUS and rel_end[0] <= 0:
                    dec_diag = True
                    rel_end = MIRROR_Y + rel_end * FLIP_Y
                    rel_vec = rel_vec * FLIP_Y
        else:  # 1 case - where there are no bricks - but should have been dealt with earlier
            raise AssertionError("Won't be seeing this.")

        logic.decrement_bricks(self.grid, seg_x, seg_y, x_flip, y_flip, dec_left, dec_diag, dec_top)
        real_end, real_vec = logic.get_rel_values(rel_end, rel_vec, x_flip, y_flip)

        self.position = real_end + offset
        self.velocity = real_vec

    @staticmethod
    def get_circle_intersections(start: npt.NDArray[float], vector: npt.NDArray[float]) \
            -> list[npt.NDArray[float], npt.NDArray[float]] | list:
        """Return all intersection points between a circle of RADIUS at the origin, and a line segment starting from
        start and ending at start + vector."""
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

    @staticmethod
    def circle_reflect(start: npt.NDArray[float], hit_point: npt.NDArray[float], vector: npt.NDArray[float]) \
            -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Given the starting position, the hit point, and a velocity,
        calculate the resulting position and velocity after a collision with a circle at the origin."""
        traveled = physics.dist(start, hit_point)
        vec_len = physics.length(vector)
        to_travel = vec_len - traveled

        half_rotation = physics.get_rotation(vector, - hit_point)
        full_rotation = half_rotation * 2
        new_vector = - physics.rotate(vector, full_rotation)
        new_pos = hit_point + new_vector * to_travel / vec_len
        return new_pos, new_vector
