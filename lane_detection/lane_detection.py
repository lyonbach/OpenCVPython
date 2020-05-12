import cv2
import numpy as np
from path import Path

current_dir = Path(__file__).parent
image_path = current_dir.parent / "samples/lane_detection.png"


def initialize_image(image):

    """
    Returns the blurred and edged version of the input image.
    """
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 0, 120)


def get_region_of_interest(image): 
    
    """
    """
    
    height = image.shape[0]
    triangle = np.array([[(315, 310), (774, 610), (20, 517)]])  # Define triangle points.
    mask = np.zeros_like(image)  # Create a black image as the same size of input image.
    cv2.fillPoly(mask, triangle, 255)
    return mask

def draw_lines(image, lines):

    """
    Draws lines on source image and returns it.
    """
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return image

# Step 1: Prepare the image.
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
initialized = initialize_image(image)
mask = get_region_of_interest(initialized)

# Step 2: Mask image to get rid of irrelevant areas.
masked = initialized &  mask

# Step 3: Create the Houghspace.
lines = cv2.HoughLinesP(masked, 2, np.pi/180.0, 100, np.array([]), minLineLength=40, maxLineGap=5)
lined = draw_lines(image.copy(), lines)

cv2.imshow("original", lined)
cv2.waitKey(0)