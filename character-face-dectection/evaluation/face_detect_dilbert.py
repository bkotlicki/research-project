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

os.chdir('../data/scraped_images_dilbert/')

for file in glob.glob("*.png"):
    img = cv2.imread(file)

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print ("Found faces: ", format(len(faces)))

    counter = 0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
