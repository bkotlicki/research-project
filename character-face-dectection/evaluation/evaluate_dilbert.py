import math
import cv2
import json
import numpy as np
import tensorflow as tf
import time

from iou import IntersectionOverUnion

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

max_val = 2 ^ 60 * (-1)
min_val = 2 ^ 60

start_time = time.time()

for k in json_dict:
    # print(k)
    img = cv2.imread('../data/resized/' + k)
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (50, 50))
        face = np.reshape(face, (1, 50, 50, 3))
        face_div = face / 255.
        prediction = face_predictor.predict(face_div)
        is_face = np.reshape(prediction, (1,))

        # print(is_face)

        if is_face < min_val:
            min_val = is_face
        if is_face > max_val:
            max_val = is_face

        # print(1 / (1 + math.exp(-is_face)))

        is_face = 1 / (1 + math.exp(-is_face))

        # cv2.imshow("", img[y:y + h, x:x + w])
        # cv2.waitKey(0)

        if is_face >= threshold:
            # cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            character_prediction = character_predictor.predict(face)
            character_prediction = np.rint(character_prediction)

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

                i = IntersectionOverUnion(min_box, [x, y, w, h])

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

            # print(character_prediction)

    # if len(cs) == len(bbs):
    #     for i in range(0, len(cs)):
    #         bb = bbs[i]
    #         face = img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    #         face = cv2.resize(face, (50, 50))
    #         face = np.reshape(face, (50, -1))
    #         face = np.reshape(face, (-1))
    #         if cs[i] != "9":
    #             images = np.append(images, face)
    #             characters = np.append(characters, int(cs[i]))

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
print("min ", min_val)
print("max", max_val)
print("\n")

time_elapsed = time.time() - start_time

print("time elapsed: ", time_elapsed)

precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1_score)
chars.append(character_recognitions)

# threshold += 0.01


print(precisions)
print(recalls)
print(f1_scores)
print(chars)
