from pathlib import Path

import cv2

if __name__ == "__main__":

    source_image_path = Path(__file__).parent / "resources" / "test_pattern.jpg"
    output_image_path = Path(__file__).parent / "resources" / "output_image.jpg"

    source_image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
    output_image = cv2.imread(str(output_image_path), cv2.IMREAD_UNCHANGED)

    diff = cv2.absdiff(source_image, output_image)

    print(source_image.shape)
    print(output_image.shape)

    cv2.imshow("Difference", diff)
    cv2.waitKey(0)

    if diff.any():
        print("Problem!!!")