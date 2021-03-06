import math
import cv2
import json
import numpy as np
import tensorflow as tf
import time

from iou import IntersectionOverUnion

path_one = '../data/resized_char_and_colour_0_750.json'
# path_one = '../data/annotations.json'

json_dict = {}

precisions = []
recalls = []
annotations = []
f1_scores = []
chars = []
times_elapsed = []

# define cnn predictors
face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_face_vggnet')
face_predictor_2 = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_recognize_human_face_2')
character_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_character_recognition')

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

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

    max_val = 2 ^ 60 * (-1)
    min_val = 2 ^ 60

    start_time = time.time()

    counter = 0

    for k in json_dict:
        img = cv2.imread('../data/resized/' + k)
        cs = json_dict[k]['Characters']
        bbs = json_dict[k]['Bounding boxes']

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([5, 110, 230])
        upper_yellow = np.array([20, 125, 250])

        dogbert_white_lower = np.array([0, 0, 150])
        dogbert_white_upper = np.array([0, 0, 254])

        catbert_red_lower = np.array([175, 200, 205])
        catbert_red_upper = np.array([180, 210, 210])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(img, img, mask=mask)

        mask_2 = cv2.inRange(hsv, dogbert_white_lower, dogbert_white_upper)
        res_2 = cv2.bitwise_and(img, img, mask=mask_2)

        mask_3 = cv2.inRange(hsv, catbert_red_lower, catbert_red_upper)
        res_3 = cv2.bitwise_and(img, img, mask=mask_3)

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
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))
            face = np.reshape(face, (1, 50, 50, 3))
            face_div = face / 255.
            prediction = face_predictor.predict(face_div)
            is_face = np.reshape(prediction, (1,))

            if is_face < min_val:
                min_val = is_face
            if is_face > max_val:
                max_val = is_face

            is_face = 1 / (1 + math.exp(-is_face))

            if is_face >= threshold:
                character_prediction = character_predictor.predict(face)
                character_prediction = np.rint(character_prediction)

                if len(bbs) > 0 and len(bbs) == len(cs):
                    which_box = 0
                    min_dist = 2 ^ 60
                    min_box = bbs[which_box]
                    for index in range(0, len(bbs)):
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
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        # true positive found
                        total_true_positives = total_true_positives + 1
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

        for i in range(0, len(bbs)):
            box = bbs[i]
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        print("tp = ", total_true_positives)
        print("fp = ", total_false_positives)
        print("gt = ", total_ground_truths)
        print("\n")

    if (total_true_positives + total_false_positives) != 0:
        precision = total_true_positives / (total_true_positives + total_false_positives)
        recall = total_true_positives / total_ground_truths
        f1_score = 2 * (precision * recall) / (precision + recall)
        character_recognitions = correct_character_detections / (
                    incorrect_character_detections + correct_character_detections)
    else:
        precision = 0
        recall = total_true_positives / total_ground_truths
        f1_score = 0
        character_recognitions = 0

    f1_score = round(f1_score, 2)

    # character_recognitions = correct_character_detections / (incorrect_character_detections + correct_character_detections)
    time_elapsed = time.time() - start_time

    print("time elapsed: ", time_elapsed)
    print("threshold: ", threshold)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1_score)
    print("character recognitions: ", character_recognitions)
    print("min ", min_val)
    print("max", max_val)
    print("\n")

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    chars.append(character_recognitions)
    times_elapsed.append(time_elapsed)

    threshold += 0.1


print(precisions)
print(recalls)
print(f1_scores)
print(chars)
print(times_elapsed)