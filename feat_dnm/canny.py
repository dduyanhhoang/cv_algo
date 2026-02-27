import numpy as np
import cv2


def get_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generates a 2D Gaussian kernel."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def detect_edges_canny(image: np.ndarray,
                       low_threshold_ratio: float = 0.05,
                       high_threshold_ratio: float = 0.15
                       ) -> np.ndarray:
    """
    Detects edges using the Canny Edge Detection algorithm from scratch.
    """
    gaussian_kernel = get_gaussian_kernel(5, 1.4)
    smoothed_img = cv2.filter2D(image.astype(np.float32), -1, gaussian_kernel)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Ix = cv2.filter2D(smoothed_img, -1, Kx)
    Iy = cv2.filter2D(smoothed_img, -1, Ky)

    magnitude = np.hypot(Ix, Iy)
    magnitude = magnitude / magnitude.max() * 255  # Normalize to 0-255
    theta = np.arctan2(Iy, Ix)

    M, N = image.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    high_threshold = Z.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_threshold)
    zeros_i, zeros_j = np.where(Z < low_threshold)
    weak_i, weak_j = np.where((Z <= high_threshold) & (Z >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                try:
                    if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                            or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                            or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (
                                    res[i - 1, j + 1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass

    return res.astype(np.uint8)
