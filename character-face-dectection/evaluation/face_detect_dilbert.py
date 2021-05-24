import math

import cv2
import os
import json

import glob

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from iou import IntersectionOverUnion

cascade_path = '../data/cascades/cascade.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


for file in glob.glob("../data/resized/*.png"):
    img = cv2.imread(file)

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_image = cv2.imread(file)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_face_16_32_64_5k')
    character_predictor = tf.keras.models.load_model('../data/cnn_saved_models/dilbert_character_recognition')

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (50, 50))
        face = np.reshape(face, (1, 50, 50, 3))
        face_div = face / 255.
        prediction = face_predictor.predict(face_div)
        is_face = np.reshape(prediction, (1,))
        if is_face >= 0.4:
            cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print(np.shape(face))
            character_prediction = character_predictor.predict(face)
            character_prediction = np.rint(character_prediction)
            print(character_prediction)
            cv2.imshow("", image[y:y + h, x:x + w])
            cv2.waitKey(0)
            # cv2.putText(image, str(round(i.result(), 3)),
            #             (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5,
            #             (0, 0, 255),
            #             2)

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
