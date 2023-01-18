from __future__ import annotations

from random import randint
from constants import *
from ball import Ball
import logic
import physics

from typing import Callable

import pygame
from pygame.locals import *


class Game:
    def __init__(self, speed: int, title: str = "Bricks", fps_cap=FPS, gui: bool = True,
                 ai_override: Callable[[Game], float] = None, block: bool = False):
        self.gui = gui
        self.ai_override = ai_override
        self.block = block

        self.gui_initialized = False

        self.fps_cap = fps_cap

        self.speed = speed
        self.interval = SPACING // self.speed  # interval between balls in frames

        self.grid = [[0] * DIM_X for _ in range(DIM_Y)]
        self.points = [[0] * DIM_X for _ in range(DIM_Y)]
        self.balls = []
        self.parked_balls = []
        self.initial_velocity = None

        self.responsive = True

        shoot_x = randint(RADIUS * 2, RES_X - RADIUS * 2)
        shoot_y = RES_Y
        self.shoot_pos = np.array([shoot_x, shoot_y])

        self.mouse_vector = np.zeros(2)
        self.mouse_clicked = False

        self.ball_count = 1
        self.balls_to_shoot = self.ball_count
        self.ball_idx = 0
        self.ball_launch_count_down = 0

        self.first_fall_time = float('inf')

        self.new_shoot_pos = None

        self.iteration = 1

        logic.rand_gen(self.grid, self.points, self.iteration)
        logic.shift_down(self.grid, self.points)

        self.message_printed = False

        self.font = None
        self.smallfont = None
        self.screen = None
        self.title = title
        self.background = None
        self.clock = None

        self.game_over = False
        self.score = 0

    def gui_initialize(self):
        # Initialise screen
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 25)
        self.smallfont = pygame.font.SysFont('Arial', 20)
        self.screen = pygame.display.set_mode((RES_X, RES_Y))
        pygame.display.set_caption(self.title)

        # Fill background
        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill(BLACK)

        # Blit everything to the screen
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

        self.clock = pygame.time.Clock()

        self.gui_initialized = True

    def flip(self):
        assert self.gui_initialized
        self.clock.tick(self.fps_cap)

        self.screen.blit(self.background, (0, 0))
        logic.draw_bricks(self.screen, self.grid, self.font)
        logic.draw_points(self.screen, self.points)
        pygame.draw.circle(self.screen, BALL_COLOR, self.shoot_pos, RADIUS)

        if self.new_shoot_pos is not None:
            pygame.draw.circle(self.screen, WHITE, self.new_shoot_pos, RADIUS)

        for ball in self.balls:
            ball.draw()

        # Top banner
        refresh_rate_render = self.font.render(str(int(self.clock.get_fps())) + ' Hz', True, RED)
        refresh_rate_rect = refresh_rate_render.get_rect()
        refresh_rate_rect.right = RES_X - 10

        self.screen.blit(refresh_rate_render, refresh_rate_rect)
        self.screen.blit(self.font.render('LEVEL: ' + str(self.iteration), True, WHITE), (10, 0))
        self.screen.blit(self.font.render('BALLS: ' + str(self.ball_count), True, WHITE), (150, 0))
        self.screen.blit(self.smallfont.render('X' + str(self.balls_to_shoot), True, WHITE),
                         (self.shoot_pos[0] + 20, RES_Y - 20))

        if self.ai_override is None:
            mouse_pos = pygame.mouse.get_pos()
            self.mouse_vector = np.array(mouse_pos) - self.shoot_pos
            self.mouse_vector = logic.clipped_direction(self.mouse_vector)

        if self.responsive:
            if self.ai_override is None:
                arrow_color = BALL_COLOR
            else:
                arrow_color = RED
            logic.draw_arrow_modified(self.screen, arrow_color, self.shoot_pos, self.mouse_vector, ARROW_MAX_LENGTH)

        pygame.display.flip()

    def tick(self):
        assert not self.game_over
        if not self.gui_initialized and self.gui:
            self.gui_initialize()

        self.mouse_clicked = False
        if self.gui:
            self.flip()
        else:
            if self.ai_override is None:
                if not self.message_printed:
                    print("I don't know what you're planning to do with no interface.")
                    print("I guess we could just stare at each other forever.")
                    self.message_printed = True

        if self.gui:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self.game_over = True
                    self.score = -1
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_clicked = True

        if self.ai_override:
            shoot_angle = self.ai_override(self)
            self.mouse_vector = physics.rotate(np.array([ARROW_MAX_LENGTH, 0]), shoot_angle)
            self.mouse_vector = logic.clipped_direction(self.mouse_vector)
            # If block == True, then a mouse click needs to happen to unblock AI.
            if not self.block or self.mouse_clicked:
                self.mouse_clicked = True

        for ball in self.balls:
            self.ball_count += ball.collected_points
            ball.collected_points = 0
            ball.tick()
        live_balls = []
        for ball in self.balls:
            if not ball.terminated:
                live_balls.append(ball)
            else:
                if EARLY_TERMINATION:
                    self.parked_balls.append(ball)
                else:
                    # grab the first ball to fall
                    if ball.frame_offset < self.first_fall_time:
                        self.first_fall_time = ball.frame_offset
                        self.new_shoot_pos = np.array([ball.position[0], RES_Y])
        self.balls = live_balls

        if self.responsive:
            if self.mouse_clicked:
                self.responsive = False
                self.ball_launch_count_down = 0
                self.initial_velocity = physics.normalize(self.mouse_vector, self.speed)
                self.new_shoot_pos = None
        else:
            if self.balls_to_shoot:
                if self.ball_launch_count_down == 0:
                    self.balls_to_shoot -= 1
                    self.balls.append(
                        Ball(self.screen, np.copy(self.shoot_pos), self.initial_velocity, self.grid, self.points,
                             self.ball_idx * self.interval, self.speed))
                    self.ball_idx += 1
                    self.ball_launch_count_down = self.interval
                self.ball_launch_count_down -= 1
            if not self.balls:
                # next level
                self.iteration += 1
                logic.rand_gen(self.grid, self.points, self.iteration)
                game_over, taken_points = logic.shift_down(self.grid, self.points)
                self.ball_count += taken_points
                self.balls_to_shoot = self.ball_count
                self.ball_idx = 0

                if EARLY_TERMINATION:
                    self.parked_balls.sort(key=lambda b: b.frame_offset)
                    first_ball_to_fall = self.parked_balls[0]
                    self.new_shoot_pos = np.array([first_ball_to_fall.position[0], RES_Y])
                    self.parked_balls = []

                self.first_fall_time = float('inf')
                self.shoot_pos = self.new_shoot_pos
                self.new_shoot_pos = None

                # pprint(brick_grid)
                if game_over:
                    pygame.quit()
                    self.game_over = True
                    self.score = self.iteration
                self.responsive = True


def main(title="Bricks", ai_override: Callable[[Game], float] = None, gui: bool = True, fps_cap=FPS, block=False,
         speed_override: bool = False) -> int:
    if speed_override:
        speed = SPEED_LIMIT
    else:
        speed = SPEED

    gameobj = Game(speed, title, fps_cap, gui, ai_override, block)

    # Event loop
    while True:
        gameobj.tick()
        if gameobj.game_over:
            return gameobj.score


if __name__ == '__main__':
    score = main()
    print('GAME OVER!')
    print('Score = {}'.format(score))
