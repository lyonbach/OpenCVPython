import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input image path.")
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file.")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Weak detections threshold.")
args = vars(ap.parse_args())

# Load model.
print("[INFO]: Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# We create the blob here.
image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# We then pass the blob to the network and obtain the face detections and predictions.
print("[INFO]: Computing object detections...")
net.setInput(blob)
detections = net.forward()

# From this point, we loop over the detections and draw boxes.
for i in range(0, detections.shape[2]):
    # filter out the weak detections.
    confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# show the output image
cv2.namedWindow("Output")
(h, w, _) = image.shape
# resized_image = cv2.resize(image, (int(w/4), int(h/4)))
cv2.imshow("Output", image)
cv2.resizeWindow("Output", 640, 480)
cv2.waitKey(0)