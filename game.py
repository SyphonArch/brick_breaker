from random import randint
from constants import *
from ball import Ball
import logic
import physics

import pygame
from pygame.locals import *


def main(breaker_override=None, draw: bool = True, fps_cap=FPS) -> int:
    if draw:
        # Initialise screen
        pygame.init()
        font = pygame.font.SysFont('Arial', 25)
        smallfont = pygame.font.SysFont('Arial', 20)
        screen = pygame.display.set_mode((RES_X, RES_Y))
        pygame.display.set_caption('Bricks')

        # Fill background
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill(BLACK)

        # Blit everything to the screen
        screen.blit(background, (0, 0))
        pygame.display.flip()

        clock = pygame.time.Clock()
    else:
        clock = None
        screen = None
        background = None
        font = None
        smallfont = None

    grid = [[0] * DIM_X for _ in range(DIM_Y)]
    points = [[0] * DIM_X for _ in range(DIM_Y)]
    balls = []
    dead_balls = []
    initial_velocity = None

    responsive = True

    shoot_x = randint(RADIUS * 2, RES_X - RADIUS * 2)
    shoot_y = RES_Y
    shoot_pos = np.array([shoot_x, shoot_y])

    mouse_vector = np.zeros(2)

    ball_count = 1
    balls_to_shoot = ball_count
    ball_idx = 0
    ball_launch_count_down = 0

    first_fall_time = float('inf')

    new_shoot_pos = None

    iteration = 1

    logic.rand_gen(grid, points, iteration)
    logic.shift_down(grid, points)

    message_printed = False

    # Event loop
    while True:
        mouse_clicked = False

        if draw:
            clock.tick(fps_cap)

            screen.blit(background, (0, 0))
            logic.draw_bricks(screen, grid, font)
            logic.draw_points(screen, points)
            pygame.draw.circle(screen, BALL_COLOR, shoot_pos, RADIUS)

            if new_shoot_pos is not None:
                pygame.draw.circle(screen, WHITE, new_shoot_pos, RADIUS)

            for ball in balls:
                ball.draw()

            # Top banner
            refresh_rate_render = font.render(str(int(clock.get_fps())) + ' Hz', True, RED)
            refresh_rate_rect = refresh_rate_render.get_rect()
            refresh_rate_rect.right = RES_X - 10

            screen.blit(refresh_rate_render, refresh_rate_rect)
            screen.blit(font.render('LEVEL: ' + str(iteration), True, WHITE), (10, 0))
            screen.blit(font.render('BALLS: ' + str(ball_count), True, WHITE), (150, 0))
            screen.blit(smallfont.render('X' + str(balls_to_shoot), True, WHITE), (shoot_pos[0] + 20, RES_Y - 20))

            if responsive:
                logic.draw_arrow_modified(screen, BALL_COLOR, shoot_pos, mouse_vector, 200)

            pygame.display.flip()

            mouse_pos = pygame.mouse.get_pos()
            mouse_vector = np.array(mouse_pos) - shoot_pos
            mouse_vector = logic.clipped_direction(mouse_vector)

            for event in pygame.event.get():
                if event.type == QUIT:
                    return -1
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_clicked = True
        else:
            if breaker_override is None:
                if not message_printed:
                    print("I don't know what you're planning to do with no interface.")
                    print("I guess we could just stare at each other forever.")
                    message_printed = True
        if breaker_override is not None:
            shoot_angle = breaker_override(grid, points, shoot_pos[0])
            mouse_vector = physics.rotate(np.array([1, 0]), shoot_angle)
            mouse_clicked = True

        for ball in balls:
            ball_count += ball.collected_points
            ball.collected_points = 0
            ball.tick()
        live_balls = []
        for ball in balls:
            if ball.position[1] < RES_Y:
                live_balls.append(ball)
            else:
                if EARLY_TERMINATION:
                    dead_balls.append(ball)
                else:
                    # grab the first ball to fall
                    if ball.frame_offset < first_fall_time:
                        first_fall_time = ball.frame_offset
                        new_shoot_pos = np.array([ball.position[0], RES_Y])
        balls = live_balls

        if responsive:
            if mouse_clicked:
                responsive = False
                ball_launch_count_down = 0
                initial_velocity = physics.normalize(mouse_vector, SPEED)
                new_shoot_pos = None
        else:
            if balls_to_shoot:
                if ball_launch_count_down == 0:
                    balls_to_shoot -= 1
                    balls.append(Ball(screen, np.copy(shoot_pos), initial_velocity, grid, points,
                                      ball_idx * INTERVAL))
                    ball_idx += 1
                    ball_launch_count_down = INTERVAL
                ball_launch_count_down -= 1
            if not balls:
                # next level
                iteration += 1
                logic.rand_gen(grid, points, iteration)
                game_over, taken_points = logic.shift_down(grid, points)
                ball_count += taken_points
                balls_to_shoot = ball_count
                ball_idx = 0

                if EARLY_TERMINATION:
                    dead_balls.sort(key=lambda b: b.frame_offset)
                    first_ball_to_fall = dead_balls[0]
                    new_shoot_pos = np.array([first_ball_to_fall.position[0], RES_Y])
                    dead_balls = []

                first_fall_time = float('inf')
                shoot_pos = new_shoot_pos
                new_shoot_pos = None

                # pprint(brick_grid)
                if game_over:
                    return iteration
                responsive = True


if __name__ == '__main__':
    score = main()
    print('GAME OVER!')
    print('Score = {}'.format(score))
