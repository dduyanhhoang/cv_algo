import numpy as np
import cv2


def compute_hog(image: np.ndarray,
                cell_size: int = 8,
                block_size: int = 2,
                bins: int = 9
                ) -> tuple:
    """
    Computes the Histogram of Oriented Gradients (HOG).

    Args:
        image: 2D Grayscale image array.
        cell_size: Size of the cell in pixels (e.g., 8x8).
        block_size: Number of cells in each direction for a block (e.g., 2x2).
        bins: Number of histogram bins for angles 0 to 180.

    Returns:
        A tuple containing:
        - The flattened HOG feature descriptor (1D NumPy array).
        - A visualization image of the HOG cells (2D NumPy array).
    """
    img_float = np.float32(image) / 255.0
    h, w = img_float.shape

    kx = np.array([[-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1], [0], [1]], dtype=np.float32)

    gx = cv2.filter2D(img_float, -1, kx)
    gy = cv2.filter2D(img_float, -1, ky)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180

    h_cells = h // cell_size
    w_cells = w // cell_size
    cell_histograms = np.zeros((h_cells, w_cells, bins), dtype=np.float32)
    angle_bin_step = 180 / bins

    for i in range(h_cells):
        for j in range(w_cells):
            cell_mag = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

            bin_indices = (cell_angle // angle_bin_step).astype(int) % bins

            for b in range(bins):
                cell_histograms[i, j, b] = np.sum(cell_mag[bin_indices == b])

    h_blocks = h_cells - block_size + 1
    w_blocks = w_cells - block_size + 1
    hog_descriptor = []

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = cell_histograms[i:i + block_size, j:j + block_size, :]
            block_vector = block.flatten()
            l2_norm = np.sqrt(np.sum(block_vector ** 2) + 1e-5)
            normalized_block = block_vector / l2_norm
            hog_descriptor.extend(normalized_block)

    hog_descriptor = np.array(hog_descriptor, dtype=np.float32)

    vis_img = np.zeros((h_cells * cell_size, w_cells * cell_size), dtype=np.uint8)
    cell_radius = cell_size // 2

    for i in range(h_cells):
        for j in range(w_cells):
            cell_hist = cell_histograms[i, j, :]
            max_val = cell_hist.max()
            if max_val == 0:
                continue
            hist_norm = cell_hist / max_val

            center_y = i * cell_size + cell_radius
            center_x = j * cell_size + cell_radius

            for b in range(bins):
                if hist_norm[b] > 0.2:
                    theta = np.radians(b * angle_bin_step + (angle_bin_step / 2))
                    dx = int(np.cos(theta + np.pi / 2) * cell_radius * hist_norm[b])
                    dy = int(np.sin(theta + np.pi / 2) * cell_radius * hist_norm[b])

                    cv2.line(vis_img, (center_x - dx, center_y - dy),
                             (center_x + dx, center_y + dy), 255, 1)

    return hog_descriptor, vis_img
