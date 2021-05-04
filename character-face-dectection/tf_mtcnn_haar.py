import tensorflow as tf
from mtcnn import MTCNN
import cv2
import os

cascade_path = "./cascades/cascade.xml"
outwards_path = "generated_images/haar_mtcnn_test"

detector = MTCNN()


for i in range (1, 2040):
    image_path = "./scraped_images/" + str(i) + ".png"
    if os.path.exists(image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        counter = 0

        for (x, y, w, h) in faces:
            face_c = img[y:y + h, x:x + w]
            for face in detector.detect_faces(face_c):
                box = face["box"]
                cv2.rectangle(face_c, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

                filename = str(i) + "face_" + str(counter) + ".png"
                counter = counter + 1
                cv2.imwrite(os.path.join(outwards_path, filename), face_c)

        # new_filename = str(i) + "_result.png"
        # cv2.imwrite(os.path.join(outwards_path, new_filename), img)
