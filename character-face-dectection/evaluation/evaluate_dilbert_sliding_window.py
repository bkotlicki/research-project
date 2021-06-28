import math
import time
import tensorflow as tf
import cv2
import imutils
import numpy as np
import json
from iou import IntersectionOverUnion
from nms import nms


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


path_one = '../data/annotations.json'

json_dict = {}

precisions = []
recalls = []
annotations = []
f1_scores = []
chars = []

# define cnn predictors
face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_face_vggnet')
character_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_character_recognition')

# define haar
cascade_path = '../data/cascades/cascade.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

with open(path_one) as out_file:
    json_dict = json.load(out_file)

threshold = 0.0

while threshold <= 1.0:
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

        counter += 1

        print(k)
        print("tp = ", total_true_positives)
        print("fp = ", total_false_positives)
        print("gt = ", total_ground_truths)

        image = cv2.imread('../data/resized/' + k)
        cs = json_dict[k]['Characters']
        bbs = json_dict[k]['Bounding boxes']

        for box in bbs:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        vcs = np.array([])

        detected_bounding_boxes = []
        scores = []

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
        (winW, winH) = (50, 50)

        for resized in pyramid(image, scale=1.1):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                clone = resized.copy()
                face = image[y:y + winH, x:x + winW]
                face = cv2.resize(face, (50, 50))
                face = np.reshape(face, (1, 50, 50, 3))
                face_div = face / 255.
                prediction = face_predictor.predict(face_div)
                is_face = np.reshape(prediction, (1,))

                is_face = 1 / (1 + math.exp(-is_face))

                factor = 256. / resized.shape[0]

                if is_face >= threshold:
                    b = (x, y, x + int(winW * factor), y + int(winH * factor))
                    detected_bounding_boxes.append(b)
                    scores.append(is_face)

        # print(detected_bounding_boxes)

        # detected_bounding_boxes = np.array(detected_bounding_boxes)

        # pick = non_max_suppression_slow(detected_bounding_boxes, 0.3)
        pick = nms.boxes(detected_bounding_boxes, scores)

        print(pick)

        # for (startX, startY, endX, endY) in pick:
        for i in pick:
            (startX, startY, endX, endY) = detected_bounding_boxes[i]
            if len(bbs) > 0 and len(bbs) == len(cs):
                which_box = 0
                min_dist = 2 ^ 60
                min_box = bbs[which_box]
                for index in range(0, len(bbs)):
                    # print("ground truth box -> ", box)
                    box = bbs[index]
                    dist = math.sqrt(pow(startX - box[0], 2) + pow(startY - box[1], 2))
                    if dist < min_dist:
                        which_box = index
                        min_dist = dist
                        min_box = box

                i = IntersectionOverUnion(min_box, [startX, startY, endX - startX, endY - startY])

                if i.result() < 0.4:
                    # false positive found
                    total_false_positives = total_false_positives + 1
                else:
                    # true positive found
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    total_true_positives = total_true_positives + 1

                    # face = image[startY:startY + endY - startY, startX:startX + endX - startX]
                    # face = cv2.resize(face, (50, 50))
                    # face = np.reshape(face, (1, 50, 50, 3))

                    # character_prediction = character_predictor.predict(face)
                    # character_prediction = np.rint(character_prediction)
                    # character_prediction = character_prediction.reshape((-1,))

                    # vcs[which_box]

                    # if int(cs[which_box]) != 9:
                    #     # print(vcs[which_box])
                    #     character_box = np.delete(vcs[which_box], 8)
                    #     # print(character_box)
                    #     # print(character_prediction)
                    #     if np.array_equal(character_box, character_prediction):
                    #         correct_character_detections += 1
                    #     else:
                    #         incorrect_character_detections += 1

            else:
                total_false_positives = total_false_positives + 1

    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / total_ground_truths

    f1_score = 2 * (precision * recall) / (precision + recall)

    f1_score = round(f1_score, 2)

    # character_recognitions = correct_character_detections / (incorrect_character_detections + correct_character_detections)

    print("threshold: ", threshold)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1_score)
    # print("character recognitions: ", character_recognitions)
    print("\n")

    time_elapsed = time.time() - start_time

    print("time elapsed: ", time_elapsed)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    # chars.append(character_recognitions)

    threshold += 0.1


print(precisions)
print(recalls)
print(f1_scores)
print(chars)
