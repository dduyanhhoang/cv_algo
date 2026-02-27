import cv2
import numpy as np


def load_image(filepath: str) -> np.ndarray:
    """
    Loads an image from the specified filepath.

    Returns the image as a NumPy array in RGB format.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale using the luminosity method.

    Formula: Grayscale = 0.299*R + 0.587*G + 0.114*B
    """
    img_float = img.astype(np.float32)

    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B

    return gray.astype(np.uint8)
