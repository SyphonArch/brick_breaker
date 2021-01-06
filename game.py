import pygame
from pygame.locals import *
from random import randint, shuffle, choice
from pprint import pprint
import math
import numpy as np
from constants import *
from ball import Ball, normalize, rotate

import pygame
from pygame.locals import *


def brick_color(value):
    redness = min(value, 15) / 15
    return 230 + redness * 25, 230 - redness * 230, 0


def gridpos_to_coordinates(i, j):
    x, y = j * WIDTH, i * HEIGHT
    return x, y


def draw_bricks(screen, grid, font):
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


def draw_points(screen, points):
    for i in range(len(points)):
        line = points[i]
        for j in range(len(line)):
            if line[j]:
                x, y = gridpos_to_coordinates(i, j)
                pygame.draw.circle(screen, GREEN, (x + WIDTH // 2, y + HEIGHT // 2), RADIUS)


def rand_gen(grid, points, n):
    brick_count = randint(MINBRICK, MAXBRICK)
    assert all(grid[0][i] == 0 for i in range(DIM_X))
    assert all(points[0][i] == 0 for i in range(DIM_X))
    placement = [n] * brick_count + [0] * (DIM_X - brick_count)
    shuffle(placement)
    grid[0] = placement
    empties = [i for i in range(len(placement)) if placement[i] == 0]
    point_idx = choice(empties)
    points[0][point_idx] = 1


def shift_down(grid, points):
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


def draw_arrow(screen, color, start, end, trirad=10, thickness=4):
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


def draw_arrow_modified(screen, color, start, vector, length):
    if not any(vector):
        return
    orig_length = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    vector *= min(length / orig_length, 1)
    end = start + vector
    draw_arrow(screen, color, start, end)


MAX_ANGLE = 180 - MIN_ANGLE
MIN_ANGLE_RAD = (180 + MIN_ANGLE) / 180 * np.pi
MAX_ANGLE_RAD = (180 + MAX_ANGLE) / 180 * np.pi


def clipped_direction(vector):
    angle = np.arctan2(vector[1], vector[0]) + np.pi * 2
    target_angle = min(MAX_ANGLE_RAD, max(MIN_ANGLE_RAD, angle))
    return rotate(vector, target_angle - angle)


def main():
    # Initialise screen
    pygame.init()
    font = pygame.font.SysFont('Arial', 25)
    smallfont = pygame.font.SysFont('Arial', 20)
    screen = pygame.display.set_mode((WIDTH * DIM_X, HEIGHT * DIM_Y))
    pygame.display.set_caption('Bricks')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill(BLACK)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

    grid = [[0] * 6 for _ in range(9)]
    points = [[0] * 6 for _ in range(9)]
    balls = []
    initial_velocity = None

    clock = pygame.time.Clock()

    responsive = True

    mouse_clicked = False

    shoot_x = randint(RADIUS * 2, WIDTH * DIM_X - RADIUS * 2)
    shoot_y = HEIGHT * DIM_Y
    shoot_pos = np.array([shoot_x, shoot_y])

    ball_count = 1
    balls_to_shoot = 0
    count_down = 0

    new_shoot_pos = None

    iteration = 1

    rand_gen(grid, points, iteration)
    game_over = shift_down(grid, points)
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
        # Top banner
        screen.blit(font.render(str(int(clock.get_fps())) + ' Hz', True, RED), (WIDTH * DIM_X - 70, 0))
        screen.blit(font.render('LEVEL: ' + str(iteration), True, WHITE), (10, 0))
        screen.blit(font.render('BALLS: ' + str(ball_count), True, WHITE), (150, 0))
        draw_bricks(screen, grid, font)
        draw_points(screen, points)
        pygame.draw.circle(screen, BALL_COLOR, tuple(shoot_pos), RADIUS)
        if new_shoot_pos is not None:
            pygame.draw.circle(screen, WHITE, tuple(new_shoot_pos), RADIUS)
        for ball in balls:
            ball_count += ball.collected_points
            ball.collected_points = 0
            ball.tick()
        live_balls = []
        for ball in balls:
            if ball.position[1] < HEIGHT * DIM_Y:
                live_balls.append(ball)
            else:
                if new_shoot_pos is None:
                    new_shoot_pos = np.array([ball.position[0], shoot_pos[1]])
        balls = live_balls
        for ball in balls:
            ball.draw()

        if responsive:
            mouse_vector = np.array(mouse_pos) - shoot_pos
            mouse_vector = clipped_direction(mouse_vector)
            draw_arrow_modified(screen, BALL_COLOR, shoot_pos, mouse_vector, 200)
            if mouse_clicked:
                responsive = False
                balls_to_shoot = ball_count
                count_down = 0
                initial_velocity = normalize(mouse_vector, SPEED)
                new_shoot_pos = None
        else:
            screen.blit(smallfont.render('X' + str(balls_to_shoot), True, WHITE),
                        (shoot_pos[0] + 20, shoot_pos[1] - 20))
            if balls_to_shoot:
                if count_down == 0:
                    balls_to_shoot -= 1
                    balls.append(Ball(screen, np.copy(shoot_pos), initial_velocity, grid, points))
                    count_down = INTERVAL
                count_down -= 1
            if not balls:
                iteration += 1
                rand_gen(grid, points, iteration)
                game_over, taken_points = shift_down(grid, points)
                ball_count += taken_points
                shoot_pos = new_shoot_pos
                new_shoot_pos = None

                # pprint(brick_grid)
                if game_over:
                    return iteration
                responsive = True

        pygame.display.flip()


if __name__ == '__main__':
    score = main()
    print('GAME OVER!')
    print('Score = {}'.format(score))
