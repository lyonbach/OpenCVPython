"""
This module is intended to be a bridge between different image processing related modules.
(opencv numpy scikit_image, etc. and consists of functions to be used as helper functions
when dealing intermediate tasks.
"""

import numpy as np
import dlib

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

