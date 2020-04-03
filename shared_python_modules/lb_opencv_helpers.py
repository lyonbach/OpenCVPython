"""
This module is intended to be a bridge between different image processing related modules.
(opencv numpy scikit_image, etc. and consists of functions to be used as helper functions
when dealing intermediate tasks.
"""

import cv2
import numpy as np
import dlib

# General.
def resize(source, width=None, height=None):

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


# dlib related.
def dlib_rectangle_to_bounding_box(rectangle):

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

