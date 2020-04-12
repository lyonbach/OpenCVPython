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

# General.
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


# dlib related.
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
