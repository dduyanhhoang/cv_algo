import numpy as np
import cv2


def detect_hough_lines(edge_image: np.ndarray,
                       original_image: np.ndarray,
                       threshold: int = 150
                       ) -> np.ndarray:
    """
    Implements the Standard Hough Transform from scratch to detect lines,
    filtered to find rectangular window structures.

    Args:
        edge_image: 2D binary/grayscale edge map (e.g., from Canny).
        original_image: RGB image to draw the detected lines on.
        threshold: Minimum number of votes in the accumulator to be considered a line.

    Returns:
        RGB image with detected horizontal and vertical lines drawn.
    """
    height, width = edge_image.shape
    max_dist = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    num_thetas = len(thetas)

    rhos = np.linspace(-max_dist, max_dist, 2 * max_dist)

    accumulator = np.zeros((2 * max_dist, num_thetas), dtype=np.int32)

    y_idxs, x_idxs = np.nonzero(edge_image)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        rho_vals = x * cos_t + y * sin_t
        rho_idxs = np.round(rho_vals).astype(int) + max_dist

        for t_idx in range(num_thetas):
            accumulator[rho_idxs[t_idx], t_idx] += 1

    output_img = original_image.copy()

    peak_rho_idxs, peak_theta_idxs = np.where(accumulator > threshold)

    for rho_idx, theta_idx in zip(peak_rho_idxs, peak_theta_idxs):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]

        angle_deg = np.rad2deg(theta)

        is_vertical = abs(angle_deg) < 15
        is_horizontal = abs(angle_deg) > 75

        if is_vertical or is_horizontal:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            color = (0, 255, 0) if is_vertical else (0, 0, 255)
            cv2.line(output_img, (x1, y1), (x2, y2), color, 2)

    return output_img
