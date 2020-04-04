import numpy as np
import cv2

def order_points(points):

	assert points.shape == (4, 1, 2), "A np.ndarray(shape = (4, 1, 2)) is required!"

	ordered_points = np.zeros((4, 1, 2), dtype=points.dtype)
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	points_sum = points.sum(axis = 2)
	ordered_points[0] = points[np.argmin(points_sum)]
	ordered_points[2] = points[np.argmax(points_sum)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	points_diff = np.diff(points, axis = 2)
	ordered_points[1] = points[np.argmin(points_diff)]
	ordered_points[3] = points[np.argmax(points_diff)]
	# return the ordered coordinates

	return ordered_points


def warp(image, points, target_height=-1, target_width=-1):

	"""
	image (np.ndarray): Image to be transformed to birds-eye-view.
	points (np.ndarray): Corner points of the deformed image.
	target_width (int): Final image width.
	target_hight (int): Final image height.
	"""

	# Obtain a consistent order (in this case ccw) of the points and unpack individually.
	source_image_rectangle = order_points(points)
	source_image_rectangle = source_image_rectangle.astype(dtype=np.float)

	(top_left, top_right, bottom_right, bottom_left) = \
		source_image_rectangle[0][0], source_image_rectangle[1][0], source_image_rectangle[2][0], source_image_rectangle[3][0]

	if target_width == -1:
		# Compute target width which is the maximum width of the deformed item* in the image if it has not been passed.
		width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
		width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
		target_width = max(int(width_a), int(width_b))

	if target_height == -1:
		# Compute target height which is the maximum height of the deformed item* in the image if it has not been passed.
		height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
		height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
		target_height = max(int(height_a), int(height_b))

	target_image_rectangle = np.array([
		[0, 0],
		[target_width - 1, 0],
		[target_width - 1, target_height - 1],
		[0, target_height - 1]], dtype="float32")
	
	# compute the perspective transform matrix and then apply it
	temp = np.zeros(shape=target_image_rectangle.shape, dtype=np.float32)
	for i in range(4):
		temp[i] = source_image_rectangle[i][0]

	source_image_rectangle = [source_image_rectangle[i] for i in range(4)]
	source_image_rectangle = temp.copy()

	M = cv2.getPerspectiveTransform(source_image_rectangle, target_image_rectangle)
	return cv2.warpPerspective(image, M, (target_width, target_height))
