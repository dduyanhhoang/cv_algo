import numpy as np


def get_translation_matrix(tx: float, ty: float) -> np.ndarray:
    """
    Constructs a 3x3 homogeneous translation matrix.

    Args:
        tx (float): Translation offset along the x-axis.
        ty (float): Translation offset along the y-axis.

    Returns:
        np.ndarray: A 3x3 transformation matrix.
    """
    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def get_rotation_matrix(angle_degrees: float) -> np.ndarray:
    """
    Constructs a 3x3 homogeneous rotation matrix for rotation around the origin.
    """
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    return np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def get_rotation_around_center_matrix(angle_degrees: float, cx: float, cy: float) -> np.ndarray:
    """
    Constructs a composite 3x3 matrix to rotate around a specific center point.
    M = T_center @ R_theta @ T_negative_center
    """
    t_to_origin = get_translation_matrix(-cx, -cy)
    rotation = get_rotation_matrix(angle_degrees)
    t_back = get_translation_matrix(cx, cy)

    return t_back @ rotation @ t_to_origin


def get_scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """
    Constructs a 3x3 homogeneous scaling matrix relative to the origin.
    """
    return np.array([
        [sx, 0.0, 0.0],
        [0.0, sy, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def get_scaling_around_center_matrix(sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    """
    Constructs a composite 3x3 matrix to scale an object around its center point.
    M = T_center @ S @ T_negative_center
    """
    t_to_origin = get_translation_matrix(-cx, -cy)
    scaling = get_scaling_matrix(sx, sy)
    t_back = get_translation_matrix(cx, cy)

    return t_back @ scaling @ t_to_origin
