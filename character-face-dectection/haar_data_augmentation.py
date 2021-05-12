import cv2
import os
import glob
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=45,
    shear_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True)

folder_path = '../../opencv_workspace/400_positives/negs/'
f = open("../../opencv_workspace/3000_positives/bg.txt", "w+")

os.chdir('../../opencv_workspace/400_positives/negs')
for file in glob.glob("*.jpg"):
    # path = '../../opencv_workspace/400_positives/negs/' + file
    img = load_img(file, color_mode="grayscale")
    # line = "negs/" + file
    # f.write(line)

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='../../3000_positives/negs',
                              # save_to_dir='preview',
                              save_prefix='image', save_format='jpeg'):
        i += 1
        if i > 15:
            break


# for i in range(0, 1000):
#     path = folder_path + str(i) + '_02.jpg'
#     img = load_img(path, color_mode="grayscale")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir='../../opencv_workspace/3000_positives/negs',
#                               # save_to_dir='preview',
#                               save_prefix='image', save_format='jpeg'):
#         i += 1
#         if i > 10:
#             break
