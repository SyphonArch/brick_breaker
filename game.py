from __future__ import annotations

from random import randint
from constants import *
from ball import Ball
import logic
import physics
import numpy as np

from typing import Callable

import pygame
from pygame.locals import *

pygame_font = None
pygame_smallfont = None
pygame_screen = None
pygame_background = None
pygame_clock = None
gui_initialized = False


class Game:
    def __init__(self, speed: int, title: str = "Bricks", fps_cap=FPS, gui: bool = True, ai_override: bool = False,
                 ai_function: Callable[[Game], float] = None, block: bool = False):
        if ai_override:
            assert ai_function is not None

        self.gui = gui
        self.ai_override = ai_override
        self.ai_function = ai_function
        self.block = block
        self.ai_keydown = False
        self.ai_shoot_angle = None

        self.fps_cap = fps_cap

        self.speed = speed
        self.interval = SPACING // self.speed  # interval between balls in frames

        self.grid = np.zeros((DIM_Y, DIM_X), dtype=int)
        self.points = np.zeros((DIM_Y, DIM_X), dtype=int)

        self.grid_before_gen = None

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

        self.title = title

        self.game_over = False
        self.score = 0

        # This is edited from outside the class, for now.
        # Should only be added after game termination, so that simulation doesn't waste time copying the history.
        self.history = None

    def gui_initialize(self):
        global pygame_font, pygame_smallfont, pygame_screen, pygame_background, pygame_clock, gui_initialized
        # Initialise screen
        pygame.init()
        pygame_font = pygame.font.SysFont('Arial', 25)
        pygame_smallfont = pygame.font.SysFont('Arial', 20)
        pygame_screen = pygame.display.set_mode((RES_X, RES_Y))
        pygame.display.set_caption(self.title)

        # Fill background
        pygame_background = pygame.Surface(pygame_screen.get_size())
        pygame_background.fill(BLACK)

        # Blit everything to the screen
        pygame_screen.blit(pygame_background, (0, 0))
        pygame.display.flip()

        pygame_clock = pygame.time.Clock()

        gui_initialized = True

    def flip(self):
        assert gui_initialized
        pygame_clock.tick(self.fps_cap)

        pygame_screen.blit(pygame_background, (0, 0))
        logic.draw_bricks(pygame_screen, self.grid, pygame_font)
        logic.draw_points(pygame_screen, self.points)
        pygame.draw.circle(pygame_screen, BALL_COLOR, self.shoot_pos, RADIUS)

        if self.new_shoot_pos is not None:
            pygame.draw.circle(pygame_screen, WHITE, self.new_shoot_pos, RADIUS)

        for ball in self.balls:
            ball.draw()

        # Top banner
        refresh_rate_render = pygame_font.render(str(int(pygame_clock.get_fps())) + ' Hz', True, RED)
        refresh_rate_rect = refresh_rate_render.get_rect()
        refresh_rate_rect.right = RES_X - 10

        pygame_screen.blit(refresh_rate_render, refresh_rate_rect)
        pygame_screen.blit(pygame_font.render('LEVEL: ' + str(self.iteration), True, WHITE), (10, 0))
        pygame_screen.blit(pygame_font.render('BALLS: ' + str(self.ball_count), True, WHITE), (150, 0))
        pygame_screen.blit(pygame_smallfont.render('X' + str(self.balls_to_shoot), True, WHITE),
                           (self.shoot_pos[0] + 20, RES_Y - 20))

        if not self.ai_override and not self.ai_keydown:
            mouse_pos = pygame.mouse.get_pos()
            self.mouse_vector = np.array(mouse_pos) - self.shoot_pos
            self.mouse_vector = logic.clipped_direction(self.mouse_vector)

        if self.responsive:
            if self.ai_override or (self.ai_keydown and self.ai_shoot_angle is not None):
                arrow_color = RED
            elif self.ai_keydown:
                arrow_color = YELLOW
            else:
                arrow_color = BALL_COLOR
            logic.draw_arrow_modified(pygame_screen, arrow_color, self.shoot_pos, self.mouse_vector, ARROW_MAX_LENGTH)

        pygame.display.flip()

    def tick(self, early_termination_override=False):
        global gui_initialized
        assert not self.game_over
        early_terminate = EARLY_TERMINATION or early_termination_override
        if not gui_initialized and self.gui:
            self.gui_initialize()

        self.mouse_clicked = False
        if self.gui:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self.game_over = True
                    self.score = -1
                    gui_initialized = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.ai_keydown = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        self.ai_keydown = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_clicked = True
            self.flip()
        else:
            if not self.ai_override:
                if not self.message_printed:
                    print("I don't know what you're planning to do with no interface.")
                    print("I guess we could just stare at each other forever.")
                    self.message_printed = True

        if self.responsive:
            if self.ai_override or self.ai_keydown:
                if self.ai_shoot_angle is None:  # If AI has not been called yet, calculate ai_shoot_angle
                    self.ai_shoot_angle = self.ai_function(self)
                self.mouse_vector = physics.rotate(np.array([ARROW_MAX_LENGTH, 0]), self.ai_shoot_angle)
                self.mouse_vector = logic.clipped_direction(self.mouse_vector)
                # If block == True, then a mouse click needs to happen to unblock AI.
                if self.ai_override:
                    if not self.block or self.mouse_clicked:
                        self.mouse_clicked = True
            if self.mouse_clicked:
                self.responsive = False
                self.ball_launch_count_down = 0
                self.initial_velocity = physics.normalize(self.mouse_vector, self.speed)
                self.new_shoot_pos = None
        else:
            for ball in self.balls:
                self.ball_count += ball.collected_points
                ball.collected_points = 0
                ball.tick()
            live_balls = []
            for ball in self.balls:
                if not ball.terminated:
                    live_balls.append(ball)
                else:
                    if early_terminate:
                        self.parked_balls.append(ball)
                    else:
                        # grab the first ball to fall
                        if ball.frame_offset < self.first_fall_time:
                            self.first_fall_time = ball.frame_offset
                            self.new_shoot_pos = np.array([ball.position[0], RES_Y])
            self.balls = live_balls
            if self.balls_to_shoot:
                if self.ball_launch_count_down == 0:
                    self.balls_to_shoot -= 1
                    self.balls.append(
                        Ball(pygame_screen, self.shoot_pos, self.initial_velocity, self.grid, self.points,
                             self.ball_idx * self.interval, self.speed, early_terminate))
                    self.ball_idx += 1
                    self.ball_launch_count_down = self.interval
                self.ball_launch_count_down -= 1
            if not self.balls:
                # next level
                self.iteration += 1

                self.grid_before_gen = self.grid.copy()

                logic.rand_gen(self.grid, self.points, self.iteration)
                game_over, taken_points = logic.shift_down(self.grid, self.points)

                self.ball_count += taken_points
                self.balls_to_shoot = self.ball_count
                self.ball_idx = 0

                if early_terminate:
                    self.parked_balls.sort(key=lambda b: b.frame_offset)
                    first_ball_to_fall = self.parked_balls[0]
                    self.new_shoot_pos = np.array([first_ball_to_fall.position[0], RES_Y])
                    self.parked_balls = []

                self.first_fall_time = float('inf')
                self.shoot_pos = self.new_shoot_pos
                self.new_shoot_pos = None
                self.mouse_vector = np.zeros(2)

                # pprint(brick_grid)
                if game_over:
                    if self.gui:
                        pygame.quit()
                        gui_initialized = False
                    self.game_over = True
                    self.score = self.iteration - 1
                self.responsive = True

    def step(self):
        """Given an input-waiting state of the game, wait for input,
        then tick until the game needs input again."""
        assert self.responsive
        # Wait for round to be fired.
        while self.responsive:
            self.tick(early_termination_override=True)
            if self.game_over:
                return
        # Wait for all balls to fall.
        while not self.responsive:
            self.tick(early_termination_override=True)
            if self.game_over:
                return
        self.ai_shoot_angle = None


def main(title="Bricks", ai_override: bool = False, ai_function: Callable[[Game], float] = None, gui: bool = True,
         fps_cap=FPS, block=False, speed_override: bool = False) -> Game:
    if speed_override:
        speed = SPEED_LIMIT
    else:
        speed = SPEED

    gamevar = Game(speed, title, fps_cap, gui, ai_override, ai_function, block)

    history = [(None, gamevar.grid.copy(), gamevar.ball_count)]

    # Event loop
    while True:
        gamevar.step()  # This is not one frame. It is one round of input. (a.k.a. step)
        history.append((gamevar.grid_before_gen, gamevar.grid.copy(), gamevar.ball_count))
        if gamevar.game_over:
            gamevar.history = history
            return gamevar


if __name__ == '__main__':
    import explorer_evaluator.explorer as explorer

    print("Let's play Brick Breaker!")
    print("You may hold [A] for AI-assist.")
    ai_callable = explorer.create_hardcoded_explorer()
    # ai_callable = explorer.create_explorer_from_gen(1)
    gameobj = main(ai_function=ai_callable, ai_override=False)
    print('GAME OVER!')
    print('Score = {}'.format(gameobj.score))
