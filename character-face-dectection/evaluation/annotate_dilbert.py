import glob

import cv2
import numpy as np
from random import randrange
import json
import os


# from https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
class BoundingBoxWidget(object):
    def __init__(self, img):
        self.original_image = cv2.imread(img)
        self.clone = self.original_image.copy()

        self.bounding_boxes = []

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

            self.bounding_boxes.append([x, y, w, h])

            cv2.imshow("image", self.clone)

            # print(face_16_32_64)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_bounding_boxes(self):
        return self.bounding_boxes

if __name__ == '__main__':

    # here define the filepath to dilbert images in your file directory
    imgs_path = '../data/scraped_images_dilbert/*'
    out_file_path = '../data/annotations.json'
    paths = glob.glob(imgs_path)
    paths = sorted(paths)
    # here define how many images you want to annotate
    selected = paths[0:2]

    result = {}

    for i, path in enumerate(selected):
        if os.path.exists(path):
            bounding_box_widget = BoundingBoxWidget(path)
            while True:
                cv2.imshow('image', bounding_box_widget.show_image())
                key = cv2.waitKey(1)

                # after annotating bounding boxes, click 'n' to do character and color annotation
                # you can also continue without annotating bounding boxes if no characters present
                if key == ord('n'):

                    print("Available characters:\n"
                          "1: Dilbert\n"
                          "2: Dogbert\n"
                          "3: Boss")

                    # what to do about non-recurring characters?
                    # building as character
                    print("Availble background colours:\n"
                          "yellow: y\n"
                          "green: g:\n"
                          "purple: p\n"
                          "blue: b\n"
                          "pink: pi\n"
                          "white:")
                    # brown??

                    char_and_colour = input(f"{i + 1}/{len(paths)}: character character etc., background_colour: ")

                    cc_list = char_and_colour.split(",")
                    chars = cc_list[0].split(" ")
                    colour = cc_list[1]

                    filename = os.path.basename(path)

                    result[filename] = {"Characters": chars, "Colour": colour, "Bounding boxes": bounding_box_widget.return_bounding_boxes()}

                    cv2.destroyAllWindows()
                    break

    with open(out_file_path, "w+") as out_file:
        json.dump(result, out_file)

    exit(1)
