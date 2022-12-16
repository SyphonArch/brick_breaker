from random import randint
from constants import *
from ball import Ball
import logic
import physics

import pygame
from pygame.locals import *


def main():
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

    grid = [[0] * DIM_X for _ in range(DIM_Y)]
    points = [[0] * DIM_X for _ in range(DIM_Y)]
    balls = []
    dead_balls = []
    initial_velocity = None

    clock = pygame.time.Clock()

    responsive = True

    mouse_clicked = False

    shoot_x = randint(RADIUS * 2, RES_X - RADIUS * 2)
    shoot_y = RES_Y
    shoot_pos = np.array([shoot_x, shoot_y])

    ball_count = 1
    balls_to_shoot = ball_count
    ball_idx = 0
    count_down = 0

    first_fall_time = float('inf')

    new_shoot_pos = None

    iteration = 1

    logic.rand_gen(grid, points, iteration)
    game_over = logic.shift_down(grid, points)
    pygame.display.flip()

    # Event loop
    while True:
        clock.tick(FPS)
        mouse_clicked = False
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_clicked = True
        mouse_pos = pygame.mouse.get_pos()

        screen.blit(background, (0, 0))
        logic.draw_bricks(screen, grid, font)
        logic.draw_points(screen, points)
        pygame.draw.circle(screen, BALL_COLOR, tuple(shoot_pos), RADIUS)
        if new_shoot_pos is not None:
            pygame.draw.circle(screen, WHITE, tuple(new_shoot_pos), RADIUS)
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
        for ball in balls:
            ball.draw()

        if responsive:
            mouse_vector = np.array(mouse_pos) - shoot_pos
            mouse_vector = logic.clipped_direction(mouse_vector)
            logic.draw_arrow_modified(screen, BALL_COLOR, shoot_pos, mouse_vector, 200)
            if mouse_clicked:
                responsive = False
                count_down = 0
                initial_velocity = physics.normalize(mouse_vector, SPEED)
                new_shoot_pos = None
        else:
            if balls_to_shoot:
                if count_down == 0:
                    balls_to_shoot -= 1
                    balls.append(Ball(screen, np.copy(shoot_pos), initial_velocity, grid, points,
                                      ball_idx * INTERVAL))
                    ball_idx += 1
                    count_down = INTERVAL
                count_down -= 1
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

        # Top banner
        refresh_rate_render = font.render(str(int(clock.get_fps())) + ' Hz', True, RED)
        refresh_rate_rect = refresh_rate_render.get_rect()
        refresh_rate_rect.right = RES_X - 10

        screen.blit(refresh_rate_render, refresh_rate_rect)
        screen.blit(font.render('LEVEL: ' + str(iteration), True, WHITE), (10, 0))
        screen.blit(font.render('BALLS: ' + str(ball_count), True, WHITE), (150, 0))
        screen.blit(smallfont.render('X' + str(balls_to_shoot), True, WHITE), (shoot_pos[0] + 20, RES_Y - 20))
        pygame.display.flip()


if __name__ == '__main__':
    score = main()
    print('GAME OVER!')
    print('Score = {}'.format(score))
