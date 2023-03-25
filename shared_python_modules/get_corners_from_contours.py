import cv2
import numpy as np

def get_corners_from_contours(contours):

    """
    Returns a numpy array of image corners from the given contours.
    """

    document_corners = np.ndarray((4, 1, 2), dtype=contours[0].dtype)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        document_corners = cv2.approxPolyDP(contour, epsilon, True)
        if len(document_corners) == 4:
            break

    return document_corners
