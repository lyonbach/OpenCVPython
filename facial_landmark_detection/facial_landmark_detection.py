import argparse
import sys

import numpy as np
import dlib
import cv2

sys.path.append("/home/lyonbach/Repositories/lbOpenCVPython/shared_python_modules")
import lb_opencv_helpers as lb_oh

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor.")
ap.add_argument("-i", "--image", required=True, help="Path to the input image.")
args = vars(ap.parse_args())

# Initialize face detector and facial landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Image read and conversions.
color = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
resized_color = lb_oh.resize(color, height=500)
resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)

# The image needs to be analysed and face must have been detected before detecting features.
detected_faces = detector(resized_gray, 1)

# Facial landmarks are detected using the face coordinates from the detection.
for i, face_rect in enumerate(detected_faces):

    # Convert the recognized face rectangle to a proper format and draw.
    (x, y, w, h) = lb_oh.dlib_rectangle_to_bounding_box(face_rect)
    cv2.rectangle(resized_color, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Write face number.
    face_string = "Face #{}".format(i)
    cv2.putText(resized_color, face_string, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

    # Determine the facial landmarks for each detected face and convert the shape to numpy array.
    shape = predictor(resized_gray, face_rect)
    shape_array = lb_oh.dlib_shape_to_np_array(shape)

    # Loop over the facial landmarks and draw circles.
    for (x, y) in shape_array:
        cv2.circle(resized_color, (x, y), 2, (0, 0, 255))

cv2.imshow("win", resized_color)
cv2.waitKey(0)
