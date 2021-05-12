# import the necessary packages
import time
import tensorflow as tf

import cv2
import imutils
import numpy as np
import json


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/face_32_64_128_25k')

json_file_path = '../data/evaluation/bounding_boxes.json'
bounding_box_data = {}

with open(json_file_path) as f:
    bounding_box_data = json.load(f)

for comic_id in bounding_box_data.keys():
    # loop over the image pyramid
    image_path = '../data/scraped_images/' + str(comic_id) + '.png'
    # load the image and define the window width and height
    image = cv2.imread(image_path)
    (winW, winH) = (50, 50)

    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            face = image[y:y + winH, x:x + winW]
            face = cv2.resize(face, (50, 50))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.reshape(face, (1, 50, 50, 1))
            face = face / 255.
            prediction = face_predictor.predict(face)

            is_face = np.reshape(prediction, (1,))

            if is_face > 0.9:
                print("face")
                cv2.imshow("Window", image[y:y + winH, x:x + winW])
                cv2.waitKey(0)
            else:
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)
