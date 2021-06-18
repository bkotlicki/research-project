import json
import math

import cv2
import time
import random
import tensorflow as tf
import numpy as np
from nms import nms
from evaluation import nms_def

from selective_search import selective_search, box_filter

def bincount_app(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


# load the input image
# image = cv2.imread('data/resized/2007-06-11_0.png')
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# ss.setBaseImage(image)
# # ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()
#
# rects = ss.process()

face_predictor = tf.keras.models.load_model('data/cnn_saved_models/dilbert_recognize_human_face_2')
character_predictor = tf.keras.models.load_model('data/cnn_saved_models/dilbert_character_recognition')
dogbert_face_predictor = tf.keras.models.load_model('data/cnn_saved_models/recognize_dogbert_3')

path_one = 'data/resized_char_and_colour_0_750.json'

json_dict = {}

with open(path_one) as out_file:
    json_dict = json.load(out_file)

# loop over the region proposals in chunks (so we can better
# visualize them)
for k in json_dict:
    if "2" not in json_dict[k]["Characters"]:
        print(k)
        # 2007-06-16_2.png
        # 2007-06-18_0.png
        # 2007-06-23_0.png
        image = cv2.imread('data/resized/' + k)

        print(image.shape)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([5, 110, 230])
        upper_yellow = np.array([20, 125, 250])

        dogbert_white_lower = np.array([0, 0, 244])
        dogbert_white_upper = np.array([0, 0, 254])

        # cv2.imshow("", hsv)
        # cv2.waitKey(0)

        catbert_red_lower = np.array([175, 200, 205])
        catbert_red_upper = np.array([180, 210, 210])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(image, image, mask=mask)

        mask_2 = cv2.inRange(hsv, dogbert_white_lower, dogbert_white_upper)
        res_2 = cv2.bitwise_and(image, image, mask=mask_2)

        mask_3 = cv2.inRange(hsv, catbert_red_lower, catbert_red_upper)
        res_3 = cv2.bitwise_and(image, image, mask=mask_3)

        # res_4 = cv2.add(res, res_2)

        combo = cv2.add(res, res_2)
        combo = cv2.add(combo, res_3)

        output = image.copy()

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
                b = [x, y, x + w, y + h]

                # color = [random.randint(0, 255) for j in range(0, 2)]
                # cv2.rectangle(res, (x, y), (x + w, y + h), color, 2)

                face = image[y:y + h, x:x + w]
                face = cv2.resize(face, (50, 50))
                face = np.reshape(face, (1, 50, 50, 3))
                face_div = face / 255.
                prediction = face_predictor.predict(face_div)
                is_face = np.reshape(prediction, (1,))

                is_face = 1 / (1 + math.exp(-is_face))

                # dominant_color = bincount_app(image[y:y + h, x:x + w])
                #
                # if (250, 250, 250) < dominant_color <= (256, 256, 256):
                #     is_dogbert = dogbert_face_predictor.predict(face_div)
                #     is_dogbert = 1 / (1 + math.exp(-is_dogbert))
                #     if is_dogbert > 0.9:
                #         detected_bounding_boxes.append(b)
                #         area = (w * h) / (8 * 8)
                #         normalized_area = 1 / (1 + math.exp(-area))
                #         scores.append(normalized_area)

                if is_face >= 0.8:
                    if b != [0, 0, 256, 256]:
                        detected_bounding_boxes.append(b)

                        area = (w * h) / (8 * 8)
                        normalized_area = 1 / (1 + math.exp(-area))

                        scores.append(normalized_area)

            # pick = nms.boxes(detected_bounding_boxes, scores, nms.malisiewicz.nms)
            print(detected_bounding_boxes)

        detected_bounding_boxes_copy = []

        for b in detected_bounding_boxes:
            t = tuple(b)
            detected_bounding_boxes_copy.append(t)

        print(detected_bounding_boxes_copy)

        detected_bounding_boxes_copy = np.array(detected_bounding_boxes_copy)

        pick = nms_def.non_max_suppression_slow(detected_bounding_boxes_copy, 0.05)

        for p in pick:
            (startX, startY, endX, endY) = p
            cv2.rectangle(output, (startX, startY), (endX, endY), (255, 0, 0), 2)

        ss.setBaseImage(res_2)

        # ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        detected_bounding_boxes = []
        scores = []

        for i in range(0, len(rects), 100):

            # loop over the current subset of region proposals
            for (x, y, w, h) in rects[i:i + 100]:
                b = [x, y, x + w, y + h]

                white_ratio = cv2.countNonZero(mask_2) / (w * h)

                # if white_ratio > 0.9:
                # color = [random.randint(0, 255) for j in range(0, 2)]
                # cv2.rectangle(res_2, (x, y), (x + w, y + h), color, 2)

                face = image[y:y + h, x:x + w]
                face = cv2.resize(face, (50, 50))
                face = np.reshape(face, (1, 50, 50, 3))
                face_div = face / 255.

                is_dogbert = dogbert_face_predictor.predict(face_div)
                is_dogbert = 1 / (1 + math.exp(-is_dogbert))

                # white_ratio = cv2.countNonZero(mask_2) / (w * h)

                if is_dogbert > 0.9:
                    detected_bounding_boxes.append(b)
                    area = (w * h) / (8 * 8)
                    normalized_area = 1 / (1 + math.exp(-area))
                    scores.append(normalized_area)

                    # cv2.imshow("", res_2[y:y + h, x:x + w])
                    # cv2.waitKey(0)

        detected_bounding_boxes_copy = []

        for b in detected_bounding_boxes:
            t = tuple(b)
            detected_bounding_boxes_copy.append(t)

        print(detected_bounding_boxes_copy)

        detected_bounding_boxes_copy = np.array(detected_bounding_boxes_copy)

        pick_2 = nms_def.non_max_suppression_slow(detected_bounding_boxes_copy, 0.05)

        for p in pick_2:
            (startX, startY, endX, endY) = p


            # is_close = False
            #
            # closest_end_x = 0
            # closest_end_y = 0
            #
            # x_left = startX
            # y_left = startY
            # x_right = endX
            # y_right = startY
            #
            # mid_point_x = x_left + (x_right - x_left) / 2
            # mid_point_y = y_left + (y_right - y_left) / 2
            #
            # for lp in pick:
            #     (sX, sY, eX, eY) = lp
            #
            #     xl = sX
            #     yl = sY
            #     xr = eX
            #     yr = eY
            #
            #     mpx = xl + (xr - xl) / 2
            #     mpy = yl + (yr - yl) / 2
            #
            #     dist = math.sqrt(pow(mid_point_x - mpx, 2) + pow(startY - eY, 2))
            #
            #     if dist < 50:
            #         is_close = True
            #
            #     print(dist)
            #     print(mid_point_x, startY)
            #     print(mpx, eY)
            #     # if dist < min_dist:
            #     #     which_box = index
            #     #     min_dist = dist
            #     #     min_box = box
            #
            # if not is_close:
            cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

            print("\n")

    # show the output image
        cv2.imshow("Output", output)
        cv2.imshow("xx", res)
        cv2.imshow("Ot", res_2)
        key = cv2.waitKey(0) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
