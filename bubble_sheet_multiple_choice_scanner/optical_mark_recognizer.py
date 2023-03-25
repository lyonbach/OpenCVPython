# This module takes an image of a multiple choice buble sheet as the input
# and gives back the grade.
import os
import sys
import time
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import cv2
import numpy as np

import shared_python_modules as spm

ANSWERS_MAPPING = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}  # ABCDE - 01234
TEMP_IMG_PATH_FOR_FASTER_TESTING = "/home/lyonbach/Repositories/OpenCVStudies/samples/omr_test_01.png"

def extract_relevant_image(bare_image_path):
    
    """
    Function extracts the area that we are interested for evaluation.
    """

    gray = cv2.imread(bare_image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TRIANGLE)
    # edged = cv2.Canny(blurred, 75, 200)

    # Find contours in the edge map, then initialize
    # the contour that corresponds to the document
    edge_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(edge_contours):
        print("[WARNING]: Could not gather contours. Passing this one.")
        return np.ndarray(shape=gray.shape, dtype=gray.dtype)
    
    # Identify relevant image corners.
    document_corners = spm.get_corners_from_contours(edge_contours)

    # Step #2: Apply a perspective transform to extract the top-down, birds-eye-view of the exam.
    # Warp image to get bird-eye-view.
    return spm.warp(blurred.copy(), document_corners)
    # return spm.warp(gray.copy(), document_corners)

def get_questions_mapping(extracted_question_area): 

    """
    Returns a dictionary containing the question number as keys, bubbles center coordinates as values.
    eg. dict = {int: (int, int)}
    """
    # Step #3: Extract the set of bubbles (i.e., the possible answer choices) from the perspective transformed exam.
    # Extract the bubbles.
    thresh = cv2.Canny(extracted_question_area, 75, 200)
    all_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in all_contours:
        
        min_x = int(np.min(contour[:, 0][:, 0]))
        max_x = int(np.max(contour[:, 0][:, 0]))
        cv2.min(contour[:, 0][:, 0])
        min_y = int(np.min(contour[:, 0][:, 1]))
        max_y = int(np.max(contour[:, 0][:, 1]))

        w = max_x - min_x
        h = max_y - min_y
        c = (int(min_x + float(w) / 2), int(min_y + float(h)/2))
        
        if (w not in range(20, 40)) or h not in range(20, 40):
            continue

        if abs(abs(max_x - min_x) - abs(max_y - min_y)) > 3:
            continue

        circles.append(c)

    # Step #4: Sort the questions/bubbles into rows.
    # Find clusters
    ys = [circle[1] for circle in circles]
    centroids = []
    for y in list(set([circle[1] for circle in circles])):
        if ys.count(y) > 2:
            centroids.append(y)

    # Create Dictionary From Clusters
    d = defaultdict(list)
    for circle in circles:
        for cen in centroids:
            if circle[1] in range(cen - 2, cen + 2):
                d[cen].append(circle)

    # Adjust dictionary.
    temp = {}
    for i, k in enumerate(sorted(d.keys())):
        temp[i] = d[k]

    return temp.copy()

def get_answer_from_one_question(extracted_question_area, question_info):

    selected = None
    bubble_radius = 20
    max_non_zero_count = -1  # We count the maximum non zero to evaluate the answer as shaded.

    for i, item in enumerate(sorted(question_info)):
        xmin = item[0] - bubble_radius
        xmax = item[0] + bubble_radius
        ymin = item[1] - bubble_radius
        ymax = item[1] + bubble_radius
        current_question_slice = extracted_question_area[ymin:ymax, xmin: xmax]
        _, thresh = cv2.threshold(current_question_slice, 0, 255, cv2.THRESH_TRIANGLE)
        current_non_zero_count = np.count_nonzero(255 - thresh)

        if current_non_zero_count > max_non_zero_count:
            selected = list("ABCDE")[i]
            max_non_zero_count = current_non_zero_count

    return selected

def get_all_answers(extracted_question_area, all_questions_info):
    
    # Step #5: Determine the marked (i.e., “bubbled in”) answer for each row.
    answers_info = defaultdict(str)
    for qidx, one_question_info in all_questions_info.items():
        answers_info[qidx] = get_answer_from_one_question(extracted_question_area, one_question_info)
    return answers_info

def get_score(answers_info):

    # Step #6: Lookup the correct answer in our answer key to determine if the user was correct in their choice.
    keys = list("ABCDE")
    correct_count = 0
    not_answered_count = 0
    for i, (k, v) in enumerate(ANSWERS_MAPPING.items()):

        ans = answers[k] or 0

        if not ans:
            not_answered_count += 1

        correct_ans = keys[v]
        print(f"question: {k}")
        print(f"selected: {ans}")
        print(f"correct : {correct_ans}")
        if ans == correct_ans:
            correct_count += 1          
    
def read_one_paper(image_path):
    
    initial_time = time.time()

    relevant_image = extract_relevant_image(image_path)
    questions_area_info = get_questions_mapping(relevant_image)
    answers_mapping = get_all_answers(relevant_image, questions_area_info)
    delta_time = time.time() - initial_time

    return answers_mapping, delta_time

def run_tests():

    question_per_paper = 5
    test_start = time.time()
    test_count = 100

    print(f"Total paper count: {test_count}")
    print(f"Start Read.")
    for i in range(test_count):
        # print(f"Evaluation started for: {image_path}")
        answers, delta_time = read_one_paper(TEMP_IMG_PATH_FOR_FASTER_TESTING)
        # print(f"Evaluated: {image_path}")
        # print(f"Elapsed: {delta_time} seconds.")
    total_test_time = time.time() - test_start

    print(f"Total papers read: {test_count}.")
    print(f"Total number of questions: {test_count * question_per_paper}")
    print(f"Elapsed Time: {total_test_time} seconds.")
    print(f"Sample Answers: {answers}")

    # cv2.imshow("win", relevant_image)
    # cv2.waitKey(0)

def run():

    run_tests()
    # test_image = "/home/lyonbach/Downloads/image0.jpeg"
    # answers_mapping, dt = read_one_paper(test_image)
    # print(answers_mapping)
    # print(dt)

if __name__ == "__main__":
    run()
# Step #7: Repeat for all questions in the exam.
# TODO: Step #8: Display the results beautifully.

# # Show info.
# total_question_count = len(ANSWERS_MAPPING)
# score = correct_count / total_question_count * 100
# print(f"Score: {score}")



# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Source image path.")
# args = vars(ap.parse_args())
# gray = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
# assert len(edge_contours) > 0, "Could not determine any contours!"



