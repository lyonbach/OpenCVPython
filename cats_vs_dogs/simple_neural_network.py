import argparse
import os

import cv2
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

from path import Path

def image_to_feature_vector(image, size=(32, 32)):
    ":image: Image matrix. (numpy.image)"
    ":size : (tuple)"
    return cv2.resize(image, size).flatten()

def describe_images(images_path):

    print("[INFO]: Describing images...")
    data   = []
    labels = []
    image_paths = Path(images_path).files()[:1000]

    for i, image_path in enumerate(image_paths):

        image = cv2.imread(image_path)
        features = image_to_feature_vector(image)

        # Show an update every 1,000th image.
        if i > 0 and i % 1000 == 0:
            # print("[INFO] processed {}/{}".format(i, len(image_paths)))
            print("[INFO] processed {:.02f}%".format(i/len(image_paths)*100.0))

        label = image_path.namebase.split('.')[0]
        data.append(features)
        labels.append(label)

    return data, labels


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset images directory.")
    ap.add_argument("-m", "--model", required=True, help="Path to output model file.")
    args = vars(ap.parse_args())

    images_path = args["dataset"]
    data, labels = describe_images(images_path)

    # scale the input image pixels to the range [0, 1], then transform
    # the labels into vectors in the range [0, num_classes] -- this
    # generates a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    data = np.array(data) / 255.0
    labels = np_utils.to_categorical(labels, 2)

    # le = LabelEncoder()
    # labels = le.fit_transform(labels)