import cv2
import os
import json
import sys
import numpy as np

import glob
import random
from PIL import Image


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)


if __name__ == '__main__':
    left_bound = 600
    right_bound = 800

    refPt = []

    faces_folder_path = '../data/dilbert_faces/images'
    json_annotation_file = '../data/dilbert_faces/annotations.json'

    array = []

    counter = 0

    for i in range(left_bound, right_bound + 1):
        # path = "./scraped_images/" + str(i) + ".png"
        image_path = random.choice(glob.glob('../data/scraped_images_dilbert/*.png'))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        refPt = []

        clone = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click)

        process_current_image = True
        while process_current_image:
            # display the image and wait for a keypress
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                annotation = input("annotate the character: ")

                array.append(annotation)

                x = refPt[0][0]
                y = refPt[0][1]
                h = refPt[1][1] - y
                w = refPt[1][0] - x

                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                filename = str(counter) + ".png"
                counter = counter + 1
                cv2.imwrite(os.path.join(faces_folder_path, filename), face)
                process_current_image = False
            elif key == ord("e"):
                with open(json_annotation_file, 'w+') as json_file:
                    json.dump(array, json_file)
                sys.exit()
            elif key == ord("n"):
                process_current_image = False
