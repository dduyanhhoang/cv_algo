import numpy as np
import cv2


def get_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generates a 2D Gaussian kernel from scratch."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def detect_harris_corners(image: np.ndarray,
                          k: float = 0.04,
                          threshold_ratio: float = 0.01
                          ) -> np.ndarray:
    """
    Detects corners using the Harris Corner Detection algorithm.

    Args:
        image: 2D NumPy array representing a grayscale image.
        k: Harris detector free parameter in the equation.
        threshold_ratio: Ratio of the max response to use as the cutoff threshold.

    Returns:
        A BGR image (NumPy array) with red circles drawn on detected corners.
    """
    img_float = image.astype(np.float32)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    Ix = cv2.filter2D(img_float, -1, Kx)
    Iy = cv2.filter2D(img_float, -1, Ky)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    gaussian_kernel = get_gaussian_kernel(size=5, sigma=1.0)
    Sxx = cv2.filter2D(Ixx, -1, gaussian_kernel)
    Syy = cv2.filter2D(Iyy, -1, gaussian_kernel)
    Sxy = cv2.filter2D(Ixy, -1, gaussian_kernel)

    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)

    threshold = threshold_ratio * R.max()
    local_max = cv2.dilate(R, np.ones((3, 3)))
    corner_mask = (R > threshold) & (R == local_max)

    output_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    y_coords, x_coords = np.where(corner_mask)

    for y, x in zip(y_coords, x_coords):
        cv2.circle(output_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    return output_img
