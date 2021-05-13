import math

import cv2
import os
import json

import glob

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from iou import IntersectionOverUnion

# print("precision = ", precision)
# print("recall = ", recall)
cascade_path = '../data/cascades/cascade.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

outwards_path = '../data/test_dilbert'

# os.chdir('../data/scraped_images_dilbert/')

for file in glob.glob("../data/scraped_images_dilbert/*.png"):
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

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (50, 50))
        face = np.reshape(face, (1, 50, 50, 3))
        face = face / 255.
        prediction = face_predictor.predict(face)
        is_face = np.reshape(prediction, (1,))
        if is_face >= 0.7:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
