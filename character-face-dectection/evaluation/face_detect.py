import math

import cv2
import os
import json

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from iou import IntersectionOverUnion

outwards_path = '../data/generated_images/test_bounding_box'

json_file_path = '../data/evaluation/bounding_boxes.json'
bounding_box_data = {}

with open(json_file_path) as f:
    bounding_box_data = json.load(f)

threshold = 0.0

minY = 1.0
maxY = 0.0
minX = 1.0
maxX = 0.0

precisions = []
recalls = []
annotations = []
f1_scores = []

cascade_path = '../data/cascades/cascade_900_positives.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/face_32_64_128_5k_1.3k')

while threshold <= 1.0:
    total_true_positives = 0
    total_false_positives = 0
    total_ground_truths = 0

    for comic_id in bounding_box_data.keys():
        image_path = '../data/scraped_images/' + str(comic_id) + '.png'
        bounding_boxes = bounding_box_data[comic_id]

        total_ground_truths = total_ground_truths + len(bounding_boxes)

        image = cv2.imread(image_path)
        haar_image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for box in bounding_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(haar_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.reshape(face, (1, 50, 50, 1))
            face = face / 255.
            prediction = face_predictor.predict(face)
            is_face = np.reshape(prediction, (1,))
            if is_face >= threshold:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                min_dist = 2 ^ 60
                min_box = bounding_boxes[0]
                for box in bounding_boxes:
                    # print("ground truth box -> ", box)
                    dist = math.sqrt(pow(x - box[0], 2) + pow(y - box[1], 2))
                    if dist < min_dist:
                        min_dist = dist
                        min_box = box

                # print("closest box = ", min_box)
                i = IntersectionOverUnion(min_box, [x, y, w, h])

                cv2.putText(image, str(round(i.result(), 3)),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2)

                if i.result() < 0.3:
                    # false positive found
                    total_false_positives = total_false_positives + 1
                    # if i.result() > 0.2:
                    #     cv2.imshow("pow pow", image)
                    #     cv2.waitKey(0)
                else:
                    # true positive found
                    total_true_positives = total_true_positives + 1

                # print("Intersection over union = ", i.result())

        # new_filename = str(comic_id) + "_result.png"
        # # cv2.imwrite(os.path.join(outwards_path, new_filename), image)
        # vis = np.concatenate((haar_image, image), axis=0)
        # cv2.imwrite(os.path.join(outwards_path, new_filename), vis)

    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / total_ground_truths

    f1_score = 2 * (precision * recall) / (precision + recall)

    f1_score = round(f1_score, 2)

    annotation = threshold

    if precision < minX:
        minX = precision
    if precision > maxX:
        maxX = precision
    if recall < minY:
        minY = recall
    if recall > maxY:
        maxY = recall

    print("threshold: ", threshold)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1_score)
    print("\n")

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    annotation = round(annotation, 1)

    annotations.append(annotation)

    threshold += 0.1

# plt.ylim(minY - 0.05, maxY + 0.05)
# plt.xlim(minX - 0.05, maxX + 0.05)
#
# plt.title("Precision vs Recall - different CNN thresholds")
#
# plt.xlabel("precision")
# plt.ylabel("recall")
#
# print(f1_scores)
#
# plt.scatter(precisions, recalls)
#
# plt.grid()
#
# for i, txt in enumerate(annotations):
#     plt.annotate(txt, (precisions[i], recalls[i]))
#
# plt.show()


print(precisions)
print(recalls)
print(f1_scores)


# for i in range (500, 600):
#     image_path = '../data/scraped_images/' + str(i) + '.png'
#     if os.path.exists(image_path):
#         cascade_path = '../data/cascades/cascade.xml'
#
#         face_cascade = cv2.CascadeClassifier(cascade_path)
#
#         # face_predictor = tf.saved_model.load(
#         #     './cnn_saved_models/face_16_32_64'
#         # )
#
#         face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/face_32_64_128_5k_1.3k')
#
#         # face_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         faces = face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )
#
#         print ("Found faces: ", format(len(faces)))
#
#         counter = 0
#         # Draw a rectangle around the faces
#         for (x, y, w, h) in faces:
#             face = image[y:y+h, x:x+w]
#             face = cv2.resize(face, (50, 50))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             face = np.reshape(face, (1, 50, 50, 1))
#             face = face / 255.
#             prediction = face_predictor.predict(face)
#             # print(np.rint(x))
#             is_face = np.reshape(prediction, (1,))
#             # print(is_face[0])
#             if is_face >= 0.7:
#                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#             # face_16_32_64 = image[y:y+h, x:x+w]
#             # filename = str(i) + "face_" + str(counter) + ".png"
#             # counter = counter + 1
#             # cv2.imwrite(os.path.join(outwards_path, filename), face_16_32_64)
#
#
#         new_filename = str(i) + "_result.png"
#         cv2.imwrite(os.path.join(outwards_path, new_filename), image)
#
# # cv2.imshow("Faces found", image)
# # cv2.waitKey(0)
