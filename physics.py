import numpy as np


def in_seg(seg_start, seg_end, point):
    x, y = point
    min_x, max_x = sorted([seg_start[0], seg_end[0]])
    min_y, max_y = sorted([seg_start[1], seg_end[1]])
    return min_x < x <= max_x and min_y < y <= max_y


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
