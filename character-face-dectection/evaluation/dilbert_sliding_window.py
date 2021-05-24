# import the necessary packages
import math
import time
import tensorflow as tf

import cv2
import imutils
import numpy as np
import json
from iou import IntersectionOverUnion


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


path_one = '../data/resized_char_and_colour_0_750.json'

json_dict = {}

precisions = []
recalls = []
annotations = []
f1_scores = []
chars = []

# define cnn predictors
face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_face_16_32_64_5k')
character_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_character_recognition')

# define haar
cascade_path = '../data/cascades/cascade.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

with open(path_one) as out_file:
    json_dict = json.load(out_file)

threshold = 0.6

# while threshold <= 1.0:
total_true_positives = 0
total_false_positives = 0
total_ground_truths = 0
correct_character_detections = 0
incorrect_character_detections = 0

start_time = time.time()

counter = 0

for k in json_dict:

    if counter == 100:
        break

    print(k)
    image = cv2.imread('../data/resized/' + k)
    cs = json_dict[k]['Characters']
    bbs = json_dict[k]['Bounding boxes']

    vcs = np.array([])

    length = 0

    if len(bbs) > 0:
        for c in cs:
            ci = int(c) - 1
            # if ci != 8:
            vc = np.zeros((9,))
            np.put(vc, [ci], [1])
            length += 1
            vcs = np.append(vcs, vc, axis=0)

        vcs = vcs.reshape((length, 9))

    total_ground_truths = total_ground_truths + len(bbs)

    # image = cv2.imread(img)
    (winW, winH) = (30, 30)

    for resized in pyramid(image, scale=1.1):
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
            face = np.reshape(face, (1, 50, 50, 3))
            face_div = face / 255.
            prediction = face_predictor.predict(face_div)
            is_face = np.reshape(prediction, (1,))

            is_face = 1 / (1 + math.exp(-is_face))

            # print(is_face)

            if is_face >= threshold:
                # cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                character_prediction = character_predictor.predict(face)
                character_prediction = np.rint(character_prediction)

                # cv2.imshow("Window", image[y:y + winH, x:x + winW])
                # cv2.waitKey(0)

                if len(bbs) > 0 and len(bbs) == len(cs):
                    which_box = 0
                    min_dist = 2 ^ 60
                    min_box = bbs[which_box]
                    for index in range(0, len(bbs)):
                        # print("ground truth box -> ", box)
                        box = bbs[index]
                        dist = math.sqrt(pow(x - box[0], 2) + pow(y - box[1], 2))
                        if dist < min_dist:
                            which_box = index
                            min_dist = dist
                            min_box = box

                    i = IntersectionOverUnion(min_box, [x, y, winW, winH])

                    if i.result() < 0.4:
                        # false positive found
                        total_false_positives = total_false_positives + 1
                    else:
                        # true positive found
                        total_true_positives = total_true_positives + 1

                        character_prediction = character_predictor.predict(face)
                        character_prediction = np.rint(character_prediction)
                        character_prediction = character_prediction.reshape((-1,))

                        # vcs[which_box]

                        if int(cs[which_box]) != 9:
                            # print(vcs[which_box])
                            character_box = np.delete(vcs[which_box], 8)
                            # print(character_box)
                            # print(character_prediction)
                            if np.array_equal(character_box, character_prediction):
                                correct_character_detections += 1
                            else:
                                incorrect_character_detections += 1

                else:
                    total_false_positives = total_false_positives + 1

precision = total_true_positives / (total_true_positives + total_false_positives)
recall = total_true_positives / total_ground_truths

f1_score = 2 * (precision * recall) / (precision + recall)

f1_score = round(f1_score, 2)

character_recognitions = correct_character_detections / (incorrect_character_detections + correct_character_detections)

print("threshold: ", threshold)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1_score)
print("character recognitions: ", character_recognitions)
print("\n")

time_elapsed = time.time() - start_time

print("time elapsed: ", time_elapsed)

precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1_score)
chars.append(character_recognitions)

# threshold += 0.1


print(precisions)
print(recalls)
print(f1_scores)
print(chars)
