import numpy as np


class Rectangle:
    def __init__(self, p1: tuple[int, int], p2: tuple[int, int]):
        x1, y1 = p1
        x2, y2 = p2

        vertices = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]

        self.points = np.array([[x, y, 1] for x, y in vertices], dtype=np.float32).T

    def apply_transform(self, transform_matrix: np.ndarray):
        """
        Applies a 3x3 transformation matrix to the rectangle's vertices.
        """
        self.points = transform_matrix @ self.points

    def get_euclidean_vertices(self) -> np.ndarray:
        w = self.points[2, :]
        x = self.points[0, :] / w
        y = self.points[1, :] / w

        euclidean = np.vstack((x, y)).T
        return np.round(euclidean).astype(int)
