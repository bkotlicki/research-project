import json
import math

import cv2
import time
import random
import tensorflow as tf
import numpy as np
from nms import nms
from evaluation.iou import IntersectionOverUnion
import time

# load the input image
# image = cv2.imread('data/resized/2007-06-11_0.png')
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# ss.setBaseImage(image)
# # ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()
#
# rects = ss.process()

face_predictor = tf.keras.models.load_model('data/cnn_saved_models/dilbert_face_vggnet_24k')
character_predictor = tf.keras.models.load_model('data/cnn_saved_models/dilbert_character_recognition')

precisions = []
recalls = []
annotations = []
f1_scores = []
chars = []
times_elapsed = []

path_one = 'data/annotations.json'

json_dict = {}

with open(path_one) as out_file:
    json_dict = json.load(out_file)

total_true_positives = 0
total_false_positives = 0
total_ground_truths = 0
correct_character_detections = 0
incorrect_character_detections = 0

start_time = time.time()

# loop over the region proposals in chunks (so we can better
# visualize them)
for k in json_dict:
    # 2007-06-16_2.png
    # 2007-06-18_0.png
    # 2007-06-23_0.png
    image = cv2.imread('data/resized/' + k)

    cs = json_dict[k]['Characters']
    bbs = json_dict[k]['Bounding boxes']

    vcs = np.array([])

    length = 0

    output = image.copy()

    if len(bbs) > 0:
        for c in cs:
            ci = int(c) - 1
            # if ci != 8:
            vc = np.zeros((9,))
            np.put(vc, [ci], [1])
            length += 1
            vcs = np.append(vcs, vc, axis=0)

        for b in bbs:
            [a, b, c, d] = b
            cv2.rectangle(output, (a - 2, b - 2), (a + c + 2, b + d + 2), (0, 255, 0), 2)

        vcs = vcs.reshape((length, 9))

    total_ground_truths = total_ground_truths + len(bbs)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([5, 110, 230])
    upper_yellow = np.array([20, 125, 250])

    dogbert_white_lower = np.array([0, 0, 150])
    dogbert_white_upper = np.array([0, 0, 254])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(image, image, mask=mask)

    mask_2 = cv2.inRange(hsv, dogbert_white_lower, dogbert_white_upper)
    res_2 = cv2.bitwise_and(image, image, mask=mask_2)

    ss.setBaseImage(res)
    # ss.switchToSelectiveSearchFast()
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

    detected_bounding_boxes = []
    scores = []

    for i in range(0, len(rects), 100):
        # clone the original image so we can draw on it
        # output = image.copy()

        # loop over the current subset of region proposals
        for (x, y, w, h) in rects[i:i + 100]:
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))
            face = np.reshape(face, (1, 50, 50, 3))
            face_div = face / 255.
            prediction = face_predictor.predict(face_div)
            is_face = np.reshape(prediction, (1,))

            is_face = 1 / (1 + math.exp(-is_face))

            if is_face >= 0.6:
                b = [x, y, x + w, y + h]
                if b != [0, 0, 256, 256]:
                    detected_bounding_boxes.append(b)
                    area = (w * h) / (8 * 8)
                    normalized_area = 1 / (1 + math.exp(-area))
                    scores.append(normalized_area)

    pick = nms.boxes(detected_bounding_boxes, scores)

    for p in pick:
        (startX, startY, endX, endY) = detected_bounding_boxes[p]
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
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

            # [a, b, c, d] = min_box
            # # print(min_box)
            #
            # cv2.rectangle(output, (a - 5, b - 5), (a + c + 5, b + d + 5), (0, 255, 0), 2)

            if i.result() < 0.4:
                # false positive found
                total_false_positives = total_false_positives + 1
                cv2.rectangle(output, (startX - 5, startY - 5), (endX + 5, endY + 5), (0, 0, 255), 2)
            else:
                # true positive found
                cv2.rectangle(output, (startX, startY), (endX, endY), (255, 0, 0), 2)
                total_true_positives = total_true_positives + 1
                print("added TRUE POS")

                face = image[startY:startY + endY - startY, startX:startX + endX - startX]
                face = cv2.resize(face, (50, 50))
                face = np.reshape(face, (1, 50, 50, 3))

                character_prediction = character_predictor.predict(face)
                character_prediction = np.rint(character_prediction)
                character_prediction = character_prediction.reshape((-1,))

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

    print(k)
    print("tp = ", total_true_positives)
    print("fp = ", total_false_positives)
    print("gt = ", total_ground_truths)
    print(len(pick))
    print("\n")

    cv2.imwrite('data/generated_images/dilbert_selective_search/' + k + '.png', output)

precision = total_true_positives / (total_true_positives + total_false_positives)
recall = total_true_positives / total_ground_truths

f1_score = 2 * (precision * recall) / (precision + recall)

f1_score = round(f1_score, 2)

character_recognitions = correct_character_detections / (incorrect_character_detections + correct_character_detections)

time_elapsed = time.time() - start_time

print("threshold: ")
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1_score)
print("character recognitions: ", character_recognitions)
print("time elapsed: ", time_elapsed)
print("\n")
