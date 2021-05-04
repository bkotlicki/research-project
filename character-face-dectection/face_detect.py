import cv2
import os

import numpy as np
import tensorflow as tf

cascade_path = "./cascades/cascade.xml"
outwards_path = "generated_images/test_cnn_0.6"

for i in range (500, 600):
    image_path = "./scraped_images/" + str(i) + ".png"
    if os.path.exists(image_path):
        cascade_path = "./cascades/cascade.xml"

        face_cascade = cv2.CascadeClassifier(cascade_path)

        # face_predictor = tf.saved_model.load(
        #     './cnn_saved_models/face_16_32_64'
        # )

        face_predictor = tf.keras.models.load_model('cnn_saved_models/face_16_32_64')

        # face_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        image = cv2.imread(image_path)
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
            face = cv2.resize(face, (50, 50))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.reshape(face, (1, 50, 50, 1))
            face = face / 255.
            prediction = face_predictor.predict(face)
            # print(np.rint(x))
            is_face = np.reshape(prediction, (1,))
            # print(is_face[0])
            if is_face >= 0.6:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # face_16_32_64 = image[y:y+h, x:x+w]
            # filename = str(i) + "face_" + str(counter) + ".png"
            # counter = counter + 1
            # cv2.imwrite(os.path.join(outwards_path, filename), face_16_32_64)


        new_filename = str(i) + "_result.png"
        cv2.imwrite(os.path.join(outwards_path, new_filename), image)

# cv2.imshow("Faces found", image)
# cv2.waitKey(0)

