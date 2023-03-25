import unittest
import cv2
from get_corners_from_contours import *
from warp_to_birds_eye_view import *

SAMPLE_IMAGE_1 = "/home/lyonbach/Repositories/OpenCVStudies/samples/omr_test_01.png"

class TestGetCornersFromContoursModule(unittest.TestCase):

    def test_get_corners_from_contours(self):

        gray = cv2.imread(SAMPLE_IMAGE_1, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        edge_contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = get_corners_from_contours(edge_contours)

        self.assertEqual(len(corners), 4)
        self.assertIsInstance(corners, np.ndarray)

class TestWarpToBirdsEyeViewModule(unittest.TestCase):

    def test_order_points(self):

        gray = cv2.imread(SAMPLE_IMAGE_1, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        edge_contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = get_corners_from_contours(edge_contours)
        ordered_points = order_points(corners)

        self.assertEqual(len(corners), len(ordered_points))
        self.assertEqual(corners.dtype, ordered_points.dtype)

    def test_warp(self):

        gray = cv2.imread(SAMPLE_IMAGE_1, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        edge_contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = get_corners_from_contours(edge_contours)
        image = cv2.imread(SAMPLE_IMAGE_1, cv2.IMREAD_UNCHANGED)
        warped = warp(image, corners)
        cv2.imshow("win", warped)
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()