import json
import math

import cv2
import time
import random
import tensorflow as tf
import numpy as np
from nms import nms

# load the input image
# image = cv2.imread('data/resized/2007-06-11_0.png')
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# ss.setBaseImage(image)
# # ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()
#
# rects = ss.process()

face_predictor = tf.keras.models.load_model('data/cnn_saved_models/dilbert_face_vggnet_24k')

path_one = 'data/resized_char_and_colour_0_750.json'

json_dict = {}

with open(path_one) as out_file:
    json_dict = json.load(out_file)

# loop over the region proposals in chunks (so we can better
# visualize them)
for k in json_dict:
    print(k)
    # 2007-06-16_2.png
    # 2007-06-18_0.png
    # 2007-06-23_0.png
    image = cv2.imread('data/resized/' + k)

    print(image[198, 38])

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
    for i in range(0, len(rects), 100):
        # clone the original image so we can draw on it
        output = image.copy()

        detected_bounding_boxes = []
        scores = []

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
                # draw the region proposal bounding box on the image
                b = [x, y, x + w, y + h]
                print(b)
                print(is_face)
                if b != [0, 0, 256, 256]:
                    detected_bounding_boxes.append(b)
                    # scores.append(is_face)

                    area = (w * h) / (8 * 8)
                    normalized_area = 1 / (1 + math.exp(-area))

                    print(normalized_area)

                    scores.append(normalized_area)

                    color = [random.randint(0, 255) for j in range(0, 2)]
                    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        pick = nms.boxes(detected_bounding_boxes, scores)

        print("\n\n")
        print(len(pick))

        for p in pick:
            print(detected_bounding_boxes[p])
            print(scores[p])
            (startX, startY, endX, endY) = detected_bounding_boxes[p]
            cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Output", output)
        # cv2.imshow('frame', hsv)
        # # cv2.imshow('mask', mask)
        # cv2.imshow('res', res)
        # # cv2.imshow('mask_2', mask_2)
        # cv2.imshow('res_2', res_2)
        key = cv2.waitKey(0) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
