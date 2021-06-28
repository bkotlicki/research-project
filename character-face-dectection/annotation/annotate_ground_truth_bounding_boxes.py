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

    json_file_path = '../data/evaluation/bounding_boxes.json'
    bounding_box_data = {}

    with open(json_file_path) as f:
        bounding_box_data = json.load(f)

    print("Comics annotated with bounding boxes: ", len(bounding_box_data))

    run_program = True

    while run_program:
        random_comic = randrange(0, 2039)

        path = '../data/scraped_images/' + str(random_comic) + '.png'

        if os.path.exists(path):
            bounding_box_widget = BoundingBoxWidget(path)
            while True:
                cv2.imshow('image', bounding_box_widget.show_image())
                key = cv2.waitKey(1)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()

                    with open(json_file_path, 'w') as f:
                        json.dump(bounding_box_data, f)

                    run_program = False
                    break
                elif key == ord('n'):

                    bb = bounding_box_widget.return_bounding_boxes()

                    if bb:
                        data = {random_comic: bb}
                        bounding_box_data.update(data)

                    random_comic = randrange(0, 2039)

                    cv2.destroyAllWindows()
                    break

    exit(1)
