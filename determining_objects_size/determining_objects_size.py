"""
A module for measuring the size of an object.
Module was created in PyImage Search practices context.
"""

import sys

import argparse
import cv2
import numpy as np

from path import Path
sys.path.append(Path(__file__).parent.parent / "shared_python_modules")

import lb_opencv_helpers as ocvh

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

    output_image = cv2.Canny(source_image, 5, 100)
    output_image = cv2.dilate(output_image, None, iterations=1)
    output_image = cv2.erode(output_image, None, iterations=1)
    return output_image

def get_length(point_A, point_B):

    px1, py1, px2, py2 = *point_A, *point_B
    return np.math.pow(np.math.pow(px1 - px2, 2) + np.math.pow(py1 - py2, 2), 0.5)


def get_pixel_metric(left_mid_point, right_mid_point, width):

    """
    Returns the pixel_metric of the object from left-most right-most points and real width of the object.
    Note that pixel_metric can also change in units according to the unit of the object width.
    """

    lmpx, lmpy, rmpx, rmpy = *left_mid_point, *right_mid_point
    object_width = np.math.pow(np.math.pow(lmpx - rmpx, 2) + np.math.pow(lmpy - rmpy, 2), 0.5)

    return width / object_width



if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="A valid path to the image is required.")
    ap.add_argument("-w", "--width", type=float, required=True, help="Comparison object widht. (top-left)(Unit: \"cm\")")
    args = vars(ap.parse_args())

    image = initialize_image(args["image"])
    edged = get_edges(image)

    # Read the image in color to use it down for visualization.
    original_image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
    contours, hieararchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_image, contours, -1, (0, 255, 255))

    sorted_contours = ocvh.sort_contours(contours)
    pixel_metric = None

    # Examining contours.
    for i, contour in enumerate(sorted_contours):

        # Filter the contours by area to avoid unnecessary noise.
        contour_area = cv2.contourArea(contour)
        if contour_area < 500:
            continue

        # Compute the rotated bounding box of each contour.
        xs = contour[:, 0][:, 0]
        ys = contour[:, 0][:, 1]
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        sorted_points = ocvh.sort_box_points(box)

        # Find midpoints.
        top_left, top_right, bot_right, bot_left = sorted_points
        left_mid_point = (int((top_left[0] + bot_left[0])/2), int((top_left[1] + bot_left[1])/2))
        right_mid_point = (int((top_right[0] + bot_right[0])/2), int((top_right[1] + bot_right[1])/2))
        top_mid_point = (int((top_left[0] + top_right[0])/2), int((top_left[1] + top_right[1])/2))
        bot_mid_point = (int((bot_left[0] + bot_right[0])/2), int((bot_left[1] + bot_right[1])/2))

        # Calculate pixel_metric
        if i == 0:
            pixel_metric = get_pixel_metric(left_mid_point, right_mid_point, args["width"])
            # pixel_metric_v = get_pixel_metric(top_mid_point, bot_mid_point, args["width"])

        # Visualization.
        # Draw box corner points.
        for point in sorted_points:
            x, y = point
            cv2.circle(original_image, (x, y), 2, (255, 0, 0), 2)

        # Draw box midpoints
        cv2.circle(original_image, (left_mid_point[0], left_mid_point[1]), 2, (255, 0, 255), 2)
        cv2.circle(original_image, (right_mid_point[0], right_mid_point[1]), 2, (255, 0, 255), 2)
        cv2.circle(original_image, (top_mid_point[0], top_mid_point[1]), 2, (0, 255, 255), 2)
        cv2.circle(original_image, (bot_mid_point[0], bot_mid_point[1]), 2, (0, 255, 255), 2)

        p1x, p1y = left_mid_point; p2x, p2y = right_mid_point
        cv2.line(original_image, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)

        p1x, p1y = top_mid_point; p2x, p2y = bot_mid_point
        cv2.line(original_image, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)

        cv2.drawContours(original_image, [box], 0, (0,0, 255), 1)


        text_color = (255, 255, 255)
        text_x, text_y = top_mid_point

        if pixel_metric:
            dimension_h = pixel_metric * get_length(left_mid_point, right_mid_point)
            dimension_h = "{:.2f} cm".format(dimension_h)
            dimension_v = pixel_metric * get_length(top_mid_point, bot_mid_point)
            dimension_v = "{:.2f} cm".format(dimension_v)
        else:
            dimension_h, dimension_v = "NA", "NA"

        cv2.putText(original_image,
                    dimension_h,
                    (text_x - 5, text_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, text_color, 1)

        text_x, text_y = right_mid_point
        cv2.putText(original_image,
                    dimension_v,
                    (text_x + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, text_color, 1)


    cv2.imshow("win", original_image)
    cv2.waitKey(0)
