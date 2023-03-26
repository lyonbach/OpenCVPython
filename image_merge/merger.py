from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":

    images_folder = Path(__file__).parent / "resources"
    images = images_folder.glob("test_pattern_*.jpg")

    sorted_images = sorted([str(item) for item in images])

    # Inputs
    list_of_images = []
    for i, image in enumerate(sorted_images):
        row = i // 3
        col = i % 3
        list_of_images.append((image, row, col))
    margin = 50 # px

    # First loop to find out how big the output will be
    total_width = total_height = 0
    max_row = max_col = -1
    image_data_type = None # While reading images we identify the image data type also
    color_count = 0
    vector_of_images = []
    for image, col, row in list_of_images:
        image_data = cv2.imread(image)
        if image_data_type is None:
            image_data_type = image_data.dtype
        if not color_count:
            color_count = image_data.shape[2]

        if row > max_row:
            max_row = row
            total_width += image_data.shape[1] - (2 * margin)

        if col > max_col:
            max_col = col
            total_height += image_data.shape[0] - (2 * margin)
        vector_of_images.append((image_data, row, col))


    print("total_width")
    print(total_width)
    print("total_height")
    print(total_height)
    # In the second loop we apply the merge
    output_image = np.zeros((total_height, total_width, color_count), dtype=image_data_type)
    current_row = current_col = 0
    for image_data, col, row in vector_of_images:
        addition_matrix = np.zeros((total_height, total_width, color_count), dtype=image_data_type)

        if col == 0:
            start_x = 0
            image_data = image_data[:, margin:]
            end_x = start_x + image_data.shape[1]
        elif col == max_col:
            image_data = image_data[:, :image_data.shape[1]-margin]
            start_x = addition_matrix.shape[1] - image_data.shape[1]
            end_x = addition_matrix.shape[1]
        elif col > current_col:
            start_x = end_x - margin * 2
            end_x = start_x + image_data.shape[1]
        current_col=col

        if row == 0:
            start_y = 0
            image_data = image_data[margin:, :]
            end_y = start_y + image_data.shape[0]
        elif row == max_row:
            image_data = image_data[:image_data.shape[0]-margin, :]
            start_y = addition_matrix.shape[0] - image_data.shape[0]
            end_y = addition_matrix.shape[0]
        elif row > current_row:
            start_y = end_y - margin * 2
            end_y = start_y + image_data.shape[0]
        current_row = row

        # Fill edition matrix
        addition_matrix[start_y:end_y, start_x:end_x] = image_data

        # Add two matrices
        output_image += addition_matrix

        # Write out the image
        cv2.imwrite(str(images_folder / "output_image.jpg"), output_image)

