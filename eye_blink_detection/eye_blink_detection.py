"""
This module is intended to cover the practice for eye blink detection from a given video.
"""

import argparse
import time
import sys

import cv2
import dlib
import numpy as np
from scipy.spatial import distance

from path import Path
sys.path.append(Path(__file__).parent.parent / "shared_python_modules")
try:
    import lb_opencv_helpers as ocvh
except ModuleNotFoundError:
    print("[ERROR]: This script should be run from the repository root directory.")
    sys.exit()


FRAME_HEIGHT = 600  #px
EYE_ASPECT_RATIO_THRESHOLD = 0.21
EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES = 3


def eye_aspect_ratio(eye) -> float:

    """
    eyes: List of coordinates of six points that are detected as a person's eye landmarks.
    """

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor.")
    ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
    args = vars(ap.parse_args())

    # Initialize dlib face detecor (HOG-based) and then create the facial landmark predictor.
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Start video stream thread.
    print("[INFO] Start reading file...")
    video_stream = cv2.VideoCapture(args["video"])

    frame_counter = 0
    blink_counter = 0
    on_blink_rect_fade_counter = 0

    while True:
        more, current_frame = video_stream.read() or np.zeros(shape=(1, 1))
        if not more:
            break

        current_frame = ocvh.resize(current_frame, height=FRAME_HEIGHT)
        current_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # current_frame_grayscale = cv2.rotate(current_frame_grayscale, cv2.ROTATE_90_CLOCKWISE)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # detect faces in the grayscale frame
        rectangles = detector(current_frame_grayscale, 0)

        for rect in rectangles:

            shape = predictor(current_frame_grayscale, rect)
            shape = ocvh.dlib_shape_to_np_array(shape)

            # shape has all facial facial landmarks. Relevant ones must be grabbed.
            left_eye_idx_start,  left_eye_idx_end  = ocvh.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            right_eye_idx_start, right_eye_idx_end = ocvh.FACIAL_LANDMARKS_68_IDXS["right_eye"]

            left_eye_landmarks  = shape[left_eye_idx_start  : left_eye_idx_end]
            right_eye_landmarks = shape[right_eye_idx_start : right_eye_idx_end]

            # Note that ear stands for eye aspect ratio.
            left_ear  = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)


            if left_ear < EYE_ASPECT_RATIO_THRESHOLD and right_ear < EYE_ASPECT_RATIO_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES:
                    on_blink_rect_fade_counter = 3
                    blink_counter += 1
                    # Draw rectengle on blink detection.
                    frame_counter = 0


            # Draw facial landmarks.
            for feature in (left_eye_landmarks, right_eye_landmarks):
                for i in range(len(feature)):
                    x1, y1 = feature[i]
                    x2, y2 = feature[i - 1]
                    cv2.line(current_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # When blink detected draw a frame around the face.
        if on_blink_rect_fade_counter > 0:
            bounding_box = ocvh.dlib_rectangle_to_bounding_box(rectangles[0])
            x, y, w, h = bounding_box
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255, 255, 255), on_blink_rect_fade_counter)



        # Write information to screen.
        info_text = []
        info_text.append("Blinks: " + str(blink_counter))
        info_text.append("EAR_L: " + str(left_ear))
        info_text.append("EAR_R: " + str(right_ear))
        info_text.append("EAR Threshold:" + str(EYE_ASPECT_RATIO_THRESHOLD))
        info_text.append("Check EAR Cons. Fr.: " + str(EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES))

        for j, info in enumerate(info_text):
            cv2.putText(current_frame, info, (10, 25+j*12), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 255, 255), 1)

        on_blink_rect_fade_counter += -1
        on_blink_rect_fade_counter = max(0, on_blink_rect_fade_counter)

        # Output information to console.
        info_text.append("#"*80)
        print("\n".join(info_text))

        cv2.imshow("Blink Count", current_frame)

        time.sleep(.015)

