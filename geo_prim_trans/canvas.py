import os
import cv2
import numpy as np


class Canvas:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.image = self._create_white_background()

    def _create_white_background(self) -> np.ndarray:
        """
        Function 1: Initialize a white background.
        Creates an (H, W, 3) tensor filled with 255.
        """
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def draw_rectangle(self, rect, color: tuple[int, int, int] = (0, 0, 0), thickness: int = 2):
        """
        Function 2: Draw the rectangle on the canvas using its vertices.
        """
        pts = rect.get_euclidean_vertices()

        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(self.image, [pts], isClosed=True, color=color, thickness=thickness)

    def save(self, filename: str):
        """
        Saves the current canvas state to the ws1 processed directory.
        """
        output_dir = os.path.join("data", "processed", "ws1")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, self.image)
        print(f"Saved: {filepath}")
