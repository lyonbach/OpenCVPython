import time

import cv2
import numpy as np
from path import Path

current_dir = Path(__file__).parent
# video_path = current_dir.parent / "samples/lane_detection.png"
video_path = "/home/lyonbach/Repositories/detecting-road-features/data/video/project_video.mp4"

FRAME_RATE = 50
FRAME_HEIGHT = 600  #px
TRIANGLE_TOP, TRIANGLE_RBOT, TRIANGLE_LBOT = (640, 420), (1100, 640), (300, 640)  # Top, Right-Bottom, Left-Bottom

def draw_limit_lines(image):

    cv2.line(image, (570, 460), TRIANGLE_LBOT, (0, 255, 0), 2)
    cv2.line(image, (720, 460), TRIANGLE_RBOT, (0, 255, 0), 2)
    return image

def average_slope_intercept(iamge, lines):
    pass

def initialize_image(image):

    """
    Returns the blurred and edged version of the input image.
    """
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 40, 120)

def get_region_of_interest(image): 
    
    """
    """
    
    height = image.shape[0]
    triangle = np.array([[TRIANGLE_TOP, TRIANGLE_RBOT, TRIANGLE_LBOT]])  # Define triangle points.
    mask = np.zeros_like(image)  # Create a black image as the same size of input image.
    cv2.fillPoly(mask, triangle, 255)
    return mask

def draw_lines(image, lines):

    """
    Draws lines on source image and returns it.
    """
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

last_frame_time = time.time()
dt = 1 / FRAME_RATE
# Initialize video.
video_stream = cv2.VideoCapture(video_path)
while True:

    more, current_frame = video_stream.read() or np.zeros(shape=(1, 1))
    if not more:
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Correct timing.
    current_time = time.time()
    dt = current_time - last_frame_time
    dt += ((1.0 / FRAME_RATE) - dt)
    # Step 1: Prepare the image.
    # image = cv2.imread(current_frame, cv2.IMREAD_UNCHANGED)
    initialized = initialize_image(current_frame)
    mask = get_region_of_interest(initialized)

    # Step 2: Mask image to get rid of irrelevant areas.
    masked = initialized &  mask

    # Step 3: Create the Houghspace.
    lines = cv2.HoughLinesP(masked, 2, np.pi/180.0, 50, np.array([]), minLineLength=30, maxLineGap=8)
    lined = current_frame.copy()
    lined = draw_limit_lines(lined)
    lined = draw_lines(lined, lines)

    time.sleep(dt)
    last_frame_time = time.time()
    cv2.imshow("win", lined)
    # cv2.imshow("mask", masked)
    # cv2.imshow("original", lined)
    # cv2.waitKey(0)