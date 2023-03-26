from pathlib import Path

import numpy as np
import cv2


if __name__ == "__main__":

    image_path = Path(__file__).parent / "resources" / "test_pattern.jpg"
    output_folder = Path(__file__).parent / "resources"
    image_path.is_file()

    image = cv2.imread(str(image_path))

    width_increment = 500
    height_increment = 300

    current_width = 0
    current_height = 0
    counter = 0
    margin = 50

    for j in range(image.shape[0] // height_increment +1):

        height_start = current_height
        height_end = current_height + height_increment if current_height + height_increment < image.shape[0] else image.shape[0]

        for i in range(image.shape[1] // width_increment + 1):

            width_start = current_width
            width_end = current_width + width_increment if current_width + width_increment < image.shape[1] else image.shape[1]

            image_height = height_end - height_start
            image_width = width_end - width_start


            output_matrix = np.zeros((image_height + 2 * margin, image_width + 2 * margin, image.shape[2]))
            # output_matrix += 100
            some_image = image.copy()[height_start:height_end, width_start: width_end]
            output_matrix[margin : margin + image_height, margin : margin + image_width] = some_image
            image_name = "test_pattern_{}.jpg".format(counter)
            image_full_path = output_folder / image_name
            cv2.imwrite(str(image_full_path), output_matrix)


            current_width += width_increment
            counter += 1

        current_height += height_increment
        current_width = 0


