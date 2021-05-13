import cv2
import numpy as np
from random import randrange
import json
import os

import glob
import random
from PIL import Image


# from https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
class BoundingBoxWidget(object):
    def __init__(self, img):
        self.original_image = cv2.imread(img)
        self.clone = self.original_image.copy()

        # self.bounding_boxes = []
        self.images = []

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))

            x = self.image_coordinates[0][0]
            y = self.image_coordinates[0][1]
            w = self.image_coordinates[1][0] - self.image_coordinates[0][0]
            h = self.image_coordinates[1][1] - self.image_coordinates[0][1]

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36, 255, 12), 2)

            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50))
            # filename = str(counter) + ".png"
            # counter = counter + 1
            # cv2.imwrite(os.path.join(faces_folder_path, filename), face)

            # self.bounding_boxes.append([x, y, w, h])

            self.images.append(face)

            cv2.imshow("image", self.clone)

            # print(face_16_32_64)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_images(self):
        return self.images

if __name__ == '__main__':

    faces_folder_path = './data/dilbert_faces/non-faces'
    json_annotation_file = './data/dilbert_faces/annotations.json'

    counter = 9

    run_program = True

    while run_program:
        path = random.choice(glob.glob('./data/scraped_images_dilbert/*.png'))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if os.path.exists(path):
            bounding_box_widget = BoundingBoxWidget(path)
            while True:
                cv2.imshow('image', bounding_box_widget.show_image())
                key = cv2.waitKey(1)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()

                    run_program = False
                    break
                elif key == ord('n'):

                    imgs = bounding_box_widget.return_images()

                    for i in imgs:
                        filename = str(counter) + ".png"
                        counter = counter + 1
                        cv2.imwrite(os.path.join(faces_folder_path, filename), i)

                    path = random.choice(glob.glob('./data/scraped_images_dilbert/*.png'))

                    cv2.destroyAllWindows()
                    break

    exit(1)