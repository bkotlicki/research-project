import glob
import os
import random

import cv2
import numpy as np


# from https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
class BoundingBoxWidget(object):
    def __init__(self, img):
        self.original_image = cv2.imread(img)
        self.clone = self.original_image.copy()

        self.ims = np.array([[]])
        self.fs = np.array([])
        self.cs = np.array([])

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
            # print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            # print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1],
            #                                           self.image_coordinates[1][0] - self.image_coordinates[0][0],
            #                                           self.image_coordinates[1][1] - self.image_coordinates[0][1]))

            x = self.image_coordinates[0][0]
            y = self.image_coordinates[0][1]
            w = self.image_coordinates[1][0] - self.image_coordinates[0][0]
            h = self.image_coordinates[1][1] - self.image_coordinates[0][1]

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36, 255, 12), 2)
            cv2.imshow("image", self.clone)

            is_face = input("annotate if it is a face (1 or 0): ")
            self.fs = np.append(self.fs, [int(is_face)])

            face = self.original_image[y:y+h, x:x+w]
            face = cv2.resize(face, (50, 50))
            face = np.reshape(face, (50, -1))
            face = np.reshape(face, (-1))

            self.ims = np.append(self.ims, [face])

            # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_results(self):
        return self.ims, self.fs

if __name__ == '__main__':
    arr_images = np.genfromtxt('./data/data_preparation/images/images_dilbert_3.csv', delimiter=',')
    arr_faces = np.genfromtxt('./data/data_preparation/annotations/faces_dilbert_3.csv', delimiter=',')

    # arr_images = np.array([])
    # arr_faces = np.array([])
    # #
    #
    arr_images_test = arr_images.reshape(arr_images.shape[0] // 7500, 50, 50, 3)
    print(len(arr_images_test))
    # #
    # x = arr_images_test[3].astype(int)
    # x = x.astype(np.uint8)

    # cv2.imshow("", x)
    # cv2.waitKey(0)
    #
    # path = random.choice(glob.glob('./data/scraped_images_dilbert/*.png'))
    # img = cv2.imread(path)
    # img = cv2.resize(img, (50, 50))
    # img = np.reshape(img, (50, -1))
    # img = np.reshape(img, (-1))
    #
    # arr_images = []
    # arr_faces = []
    #
    # arr_images = [img, img]
    #
    # np.savetxt('./data/data_preparation/images/images_dilbert.csv', arr_images, delimiter=",")

    run_program = True
    while run_program:
        path = random.choice(glob.glob('./data/scraped_images_dilbert/*.png'))
        img = cv2.imread(path)

        boundingbox_widget = BoundingBoxWidget(path)
        while True:
            cv2.imshow('image', boundingbox_widget.show_image())
            key = cv2.waitKey(1)

            # Close program with keyboard 'q'
            if key == ord('q'):
                i, f = boundingbox_widget.return_results()

                arr_images = np.append(arr_images, i)

                arr_faces = np.append(arr_faces, f)

                cv2.destroyAllWindows()

                np.savetxt('./data/data_preparation/images/images_dilbert.csv', arr_images, delimiter=",")
                np.savetxt('./data/data_preparation/annotations/faces_dilbert.csv', arr_faces, delimiter=",")

                exit(1)
            elif key == ord('n'):
                i, f = boundingbox_widget.return_results()

                arr_images = np.append(arr_images, i)

                arr_faces = np.append(arr_faces, f)

                print(arr_images.shape)

                print(arr_images.shape)

                cv2.destroyAllWindows()

                np.savetxt('./data/data_preparation/images/images_dilbert_3.csv', arr_images, delimiter=",")
                np.savetxt('./data/data_preparation/annotations/faces_dilbert_3.csv', arr_faces, delimiter=",")

                break
