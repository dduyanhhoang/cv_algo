from feat_dnm.canny import detect_edges_canny
from feat_dnm.harris import detect_harris_corners
from feat_dnm.hog import compute_hog
from feat_dnm.hough import detect_hough_lines
from geo_prim_trans.canvas import Canvas
from geo_prim_trans.geometry import Rectangle
from geo_prim_trans.transformations import (get_translation_matrix,
                                            get_rotation_around_center_matrix,
                                            get_scaling_around_center_matrix)
from utils import load_image, rgb_to_grayscale
import cv2
import os


def run_ws1():
    """
    Implementation of WS1 could be found in geo_prim_trans package.
    """
    print("Running ws 1.")
    print("1. Creating a canvas and drawing an initial rectangle.")
    canvas = Canvas(width=800, height=600)

    p1 = (100, 100)
    p2 = (300, 200)
    rect = Rectangle(p1, p2)

    canvas.draw_rectangle(rect, color=(255, 0, 0), thickness=3)
    canvas.save("1_initial_rectangle.jpg")

    print("2. Applying translation transformation to the rectangle.")
    tx, ty = 150, 200
    translation_matrix = get_translation_matrix(tx, ty)

    rect.apply_transform(translation_matrix)

    canvas2 = Canvas(width=800, height=600)
    canvas2.draw_rectangle(Rectangle(p1, p2), color=(200, 200, 200), thickness=1)
    canvas2.draw_rectangle(rect, color=(0, 255, 0), thickness=3)

    canvas2.save("2_translated_rectangle.jpg")

    print("3. Applying rotation transformation to the rectangle.")
    # 1. Calculate the center of the current rectangle
    # 2. Get the composite rotation matrix (e.g., rotate 45 degrees clockwise)
    pts = rect.get_euclidean_vertices()
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    angle = 45.0
    rotation_matrix = get_rotation_around_center_matrix(angle, cx, cy)

    rect.apply_transform(rotation_matrix)

    canvas3 = Canvas(width=800, height=600)
    canvas3.draw_rectangle(Rectangle(p1, p2), color=(200, 200, 200), thickness=1)
    canvas3.draw_rectangle(rect, color=(0, 0, 255), thickness=3)

    canvas3.save("3_rotated_rectangle.jpg")

    print("4. Applying scaling transformation to the rectangle.")
    # 1. Calculate the center of the rectangle
    # 2. Get the composite scaling matrix (Scale up by 1.5x uniformly)
    pts = rect.get_euclidean_vertices()
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    sx, sy = 1.5, 1.5
    scaling_matrix = get_scaling_around_center_matrix(sx, sy, cx, cy)

    rect.apply_transform(scaling_matrix)

    canvas4 = Canvas(width=800, height=600)
    canvas4.draw_rectangle(Rectangle(p1, p2), color=(200, 200, 200), thickness=1)
    canvas4.draw_rectangle(rect, color=(255, 0, 255), thickness=3)

    canvas4.save("4_scaled_rectangle.jpg")


def run_ws2():
    """
    Implementation of WS2 could be found in feat_dnm package.
    """
    print("Running ws 2.")
    image_path = "data/raw/building.png"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    try:
        original_img = load_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    gray_img = rgb_to_grayscale(original_img)

    print("1. Running Harris Corner Detector.")
    harris_result = detect_harris_corners(gray_img)
    cv2.imwrite(os.path.join(output_dir, "1_harris_output.jpg"), harris_result)

    print("2. Running HOG Descriptor.")
    hog_descriptor, hog_vis = compute_hog(gray_img)
    cv2.imwrite(os.path.join(output_dir, "2_hog_visualization.jpg"), hog_vis)

    print("3. Running Canny Edge Detector.")
    canny_result = detect_edges_canny(gray_img)
    cv2.imwrite(os.path.join(output_dir, "3_canny_output.jpg"), canny_result)

    print("4. Running Hough Transform (Rectangle Detection).")
    bgr_original = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    hough_result = detect_hough_lines(canny_result, bgr_original, threshold=100)
    cv2.imwrite(os.path.join(output_dir, "4_hough_output.jpg"), hough_result)

    print(f"Processing complete. Check the '{output_dir}' folder for results.")


if __name__ == "__main__":
    run_ws1()
    # run_ws2()
