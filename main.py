from feat_dnm.canny import detect_edges_canny
from feat_dnm.harris import detect_harris_corners
from feat_dnm.hog import compute_hog
from feat_dnm.hough import detect_hough_lines
from utils import load_image, rgb_to_grayscale
import cv2
import os


def main():
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
    main()
