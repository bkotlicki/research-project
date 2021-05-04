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

            is_face = input("annotate if it is a face_16_32_64 (1 or 0): ")
            self.fs = np.append(self.fs, [int(is_face)])

            face = self.original_image[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (50, 50))

            # print(face_16_32_64)

            face = np.reshape(face, (-1))
            if int(is_face) == 1:
                character = input("annotate the character: ")
                self.cs = np.append(self.cs, [int(character)])
            else:
                self.cs = np.append(self.cs, [-1])
            self.ims = np.append(self.ims, [face])

            # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_results(self):
        return self.ims, self.fs, self.cs

if __name__ == '__main__':
    arr_images = np.genfromtxt('./data_preparation/images/images.csv', delimiter=',')
    arr_face_annotations = np.genfromtxt('./data_preparation/annotations/is_face.csv', delimiter=',')
    arr_char_annotations = np.genfromtxt('./data_preparation/annotations/character.csv', delimiter=',')

    # print(arr_images)

    print(len(arr_images))
    print(len(arr_face_annotations))

    outward_images_path = './data_preparation/images/image_files'

    for i in range(550, 700):
        path = "./scraped_images/" + str(i) + ".png"
        boundingbox_widget = BoundingBoxWidget(path)
        while True:
            cv2.imshow('image', boundingbox_widget.show_image())
            key = cv2.waitKey(1)

            # Close program with keyboard 'q'
            if key == ord('q'):
                i, f, c = boundingbox_widget.return_results()

                arr_images = np.append(arr_images, i)

                arr_images = np.reshape(arr_images, (-1, 2500))

                arr_face_annotations = np.append(arr_face_annotations, f)
                arr_char_annotations = np.append(arr_char_annotations, c)

                cv2.destroyAllWindows()

                # print(arr_images)

                np.savetxt('./data_preparation/images/images.csv', arr_images, delimiter=",")
                np.savetxt('./data_preparation/annotations/is_face.csv', arr_face_annotations, delimiter=",")
                np.savetxt('./data_preparation/annotations/character.csv', arr_char_annotations, delimiter=",")

                exit(1)
            elif key == ord('n'):
                i, f, c = boundingbox_widget.return_results()

                arr_images = np.append(arr_images, i)

                arr_images = np.reshape(arr_images, (-1, 2500))

                arr_face_annotations = np.append(arr_face_annotations, f)
                arr_char_annotations = np.append(arr_char_annotations, c)

                cv2.destroyAllWindows()

                # print(arr_images)

                np.savetxt('./data_preparation/images/images.csv', arr_images, delimiter=",")
                np.savetxt('./data_preparation/annotations/is_face.csv', arr_face_annotations, delimiter=",")
                np.savetxt('./data_preparation/annotations/character.csv', arr_char_annotations, delimiter=",")

                break
