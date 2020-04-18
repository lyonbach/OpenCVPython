"""
Module for my pyimagesearch.com studies.
"""

import argparse
from collections import deque
import time
import sys

import cv2
import numpy as np
from path import Path
sys.path.append(Path(__file__).parent.parent / "shared_python_modules")

import lb_opencv_helpers as ocvh


GREEN_LOWER = (29, 86, 6)
GREEN_UPPER = (64, 255, 255)


def init_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file", required=True)
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    return vars(ap.parse_args())


if __name__ == "__main__":
    
    args = init_args()
    video_stream = cv2.VideoCapture(args["video"])

    time.sleep(2.0)  # This is an allowance for stream to load.
    points = deque(maxlen=args["buffer"])

    # Actual tracking
    while True:

        _, current_frame = video_stream.read()

        if not current_frame.any():
            break

        current_frame = ocvh.resize(current_frame, width=600)
        original_frame = current_frame.copy()
        current_frame = cv2.GaussianBlur(current_frame, (11, 11), 0)
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Add mask, erosion and dilate to remove any small other blobs that might cause tracking issues.
        mask = cv2.inRange(current_frame_hsv, GREEN_LOWER, GREEN_UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:  # Display a message if object can not be tracked.
            cv2.putText(original_frame, "OBJECT LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

        for contour in contours:  # Determine object boundaries.

            # Calculate object center.
            p1, p2 = ocvh.get_bounding_box_from_contour(contour)
            mid_point = int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)
            
            # A quick filter to avoid confusion in number of objects.
            if abs(p1[0] - p2[0]) < 20 or abs(p1[1] - p2[1]) < 20:
                continue

            points.append(mid_point)

            cv2.rectangle(original_frame, p1, p2, (0, 0, 255), 2)
            cv2.circle(original_frame, mid_point, 3, (255, 0, 0), 3)


        for i in range(len(points) - 2, -1, -1):

            speed = np.math.pow(np.math.pow(points[i][0] - points[i+1][0], 2) + np.math.pow(points[i][1] - points[i+1][1], 2), .5)

            if speed > 0:
                line_thickness = 24 - speed
                line_thickness = max(1, line_thickness)
                cv2.line(original_frame, points[i], points[i+1], (0, 255, 255, 125), int(line_thickness))

        cv2.imshow("win_original", original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(.05)
