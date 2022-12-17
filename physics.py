import numpy as np
import numpy.typing as npt


def in_seg(seg_start: npt.NDArray[float], seg_end: npt.NDArray[float], point: npt.NDArray[float]) -> bool:
    """Given a line segment and a point, return whether the point is on the segment.

    WARNING: does not check if the point is on the line formed by extending the segment.
    tbh, I don't know why I wrote this function like this, but it's working.
    """
    x, y = point
    min_x, max_x = sorted([seg_start[0], seg_end[0]])
    min_y, max_y = sorted([seg_start[1], seg_end[1]])
    return min_x < x <= max_x and min_y < y <= max_y


def get_rotation(vec1: npt.NDArray[float], vec2: npt.NDArray[float]) -> float:
    """Given two vectors, calculate the signed rotation to go from vec1 to vec2."""
    x1, y1 = vec1
    x2, y2 = vec2
    return np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)


def rotate(vector: npt.NDArray[float], angle: float) -> npt.NDArray[int]:
    """Rotate the given vector by the given angle, and return the result."""
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rot_mat @ vector


def dist(point1: npt.NDArray[float], point2: npt.NDArray[float]) -> float:
    """Return euclidean distance between two points."""
    return length(point1 - point2)


def length(vector: npt.NDArray[float]) -> float:
    """Return the length of the given vector."""
    return np.linalg.norm(vector)


def normalize(vector: npt.NDArray[float], size: int | float) -> npt.NDArray[float]:
    """Return normalized vector."""
    return vector * size / length(vector)
