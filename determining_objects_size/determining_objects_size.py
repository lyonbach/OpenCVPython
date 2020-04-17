"""
A module for measuring the size of an object.
Module was created in PyImage Search practices context.
"""

import argparse
import cv2
import numpy as np

def midpoint(p1, p2) -> tuple:

    """
    Returns coordinates of the point which lies on the middle of the line that passes accross two points.
    """

    return ((p1[0] + p2[0]) * .5, (p1[1] + p2[1]) * .5)

def initialize_image(image_path):

    """
    Returns the image that have been converted to grayscale and applied blur.
    """

    color = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)

def get_edges(source_image):

    output_image = cv2.Canny(source_image, 75, 100)
    output_image = cv2.dilate(output_image, None, iterations=1)
    output_image = cv2.erode(output_image, None, iterations=1)
    return output_image

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="A valid path to the image is required.")
    ap.add_argument("-w", "--width", type=float, required=True, help="Comparison object widht. (top-left)(Unit: \"cm\")")
    args = vars(ap.parse_args())

    image = initialize_image(args["image"])
    edged = get_edges(image)

    original_image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)

    contours, hieararchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_image, contours, -1, (0, 255, 255))

    # Examining contours.
    for contour in contours:

        # Filter them by area to avoid unnecessary noise.
        contour_area = cv2.contourArea(contour)
        if contour_area < 500:
            continue

        # Compute the rotated bounding box of each contour.
        box = cv2.minAreaRect(contour)

        xs = contour[:, 0][:, 0]
        ys = contour[:, 0][:, 1]
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        cv2.rectangle(original_image, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 1)

    cv2.imshow("win", original_image)
    cv2.waitKey(0)


# Step1: Calibration
#   - Reference Object: An object of that we know the exact dimensions in a measurable
#     unit (mm, cm, etc.) also easily locatable in the image.

#   - pixels_metric = object_width / real_width
#   eg.) us quarter:
#           real width = .955 inch
#           object_width = 150 px
#           pixels_metric = .955 / 150