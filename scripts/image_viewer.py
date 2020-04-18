# A very simple script for viewing images.

import argparse
import sys

import cv2
from path import Path

sys.path.append(Path(__file__).parent.parent / "shared_python_modules")  # Hardcoded!!!

import lb_opencv_helpers as ocvh

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Image full path to view.")
    args = vars(ap.parse_args())
    
    image_path = Path(args["image"])
    if not image_path.isfile():
        raise FileExistsError ("Given path is not a valid image! Aborted.")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # The image should always fit the screen. For this reason it is scaled.
    image = ocvh.resize(image, height=500)

    # Show image.
    window_name = image_path.basename().split(".")[0]
    cv2.imshow(window_name, image)
    while True:
        key = cv2.waitKey(0)
        if key == 113 or key == 27:
            break
    
