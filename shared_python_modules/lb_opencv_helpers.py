"""
This module is intended to be a bridge between different image processing related modules.
(opencv numpy scikit_image, etc. and consists of functions to be used as helper functions
when dealing intermediate tasks.
"""

import sys

import cv2
import dlib
import numpy as np
from path import Path

ALLOWED_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tiff")
TOP_TO_BOTTOM = 0
BOTTOM_TO_TOP = 1
LEFT_TO_RIGHT = 2
RIGHT_TO_LEFT = 3

# General
def loop_images_in_folder(path_):

    """
    Searches for images of extension in allowed extensions within the given path and shows them
    in a 500px scaled to height window.
    """

    path_ = Path(path_)

    if not path_.isdir():
        print("[lb_helpers][ERROR]: Given path is not valid!")
        return

    for ext in ALLOWED_IMAGE_EXTENSIONS:

        images_for_extension = list(path_.walkfiles(ext))

        for i, image_path in enumerate(images_for_extension):
            print("[lb_helpers][INFO]: #{}) {} of total {} images.".format(i+1, image_path, len(images_for_extension)))
            print(image_path.abspath())
            color = cv2.imread(image_path.abspath(), cv2.IMREAD_UNCHANGED)
            color_scaled = resize(color, height=500)
            window = "#{} ext: {}".format(str(i+1), ext)
            cv2.imshow(window, color_scaled)
            key = cv2.waitKey(0)

            if key == 113:
                return

            cv2.destroyWindow(window)


def sort_box_points(box):

    """
    Returns top-left, top-right, bottom-right, bottom-left points from the given box respectively.
    """

    x_sorted_box = box[np.argsort(box[:, 0]), :]

    left_most, right_most = x_sorted_box[:2], x_sorted_box[2:]
    top_left = left_most[np.argsort(left_most[:, 1])][0]
    top_right = right_most[np.argsort(right_most[:, 1])][0]
    bot_left = left_most[np.argsort(left_most[:, 1])][1]
    bot_right = right_most[np.argsort(right_most[:, 1])][1]
    return top_left, top_right, bot_right, bot_left


def sort_contours(contours, method="left-to-right") -> list:

    """
    Returns the sorted contours as list.
    """

    # Initialize the reverse flag and sort index
    reverse = False
    i = 0

    if method == RIGHT_TO_LEFT or method == BOTTOM_TO_TOP:
        reverse = True

    if method in (TOP_TO_BOTTOM, BOTTOM_TO_TOP):
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return contours


    # return sorted_contours


def get_bounding_box_from_contour(contour) -> tuple:

    """
    Returns the boundingbox top-left, and bottom-right points from the given contour.
    """

    error_message = "[ERROR]: Contour shape must be (#, 1, 2)"
    assert len(contour.shape) == 3, error_message
    assert contour.shape[1] == 1, error_message
    assert contour.shape[2] == 2, error_message

    x_values = contour[:, 0][:, 0]
    y_values = contour[:, 0][:, 1]

    x1 = np.min(x_values); x2 = np.max(x_values)
    y1 = np.min(y_values); y2 = np.max(y_values)

    return (x1, y1), (x2, y2)


# Transformation
def resize(source, width=None, height=None) -> np.ndarray:

    if not (width or height):
        print("[lb_helpers][ERROR]: Width or height needed!")
        return

    if not width or not height:
        source_height, source_width = source.shape[:2]

        if width:
            height = source_height * width / source_width

        else:
            width = source_width * height / source_height

    return cv2.resize(source, (int(width), int(height)))


# dlib related
def dlib_rectangle_to_bounding_box(rectangle) -> tuple:

    """
    Takes rectangle object and returns OpenCV style bounding box coordinates.
    :rectangle: dlib.rect object
    """

    x = rectangle.left()
    y = rectangle.top()
    w = rectangle.right() - x
    h = rectangle.bottom() - y

    return (x, y, w, h)


def dlib_shape_to_np_array(shape, dtype="int") -> np.ndarray:

    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates
