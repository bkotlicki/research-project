from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1))
# Loading a sample image
arr_images = np.genfromtxt('./data/data_preparation/images/images_test_2.csv', delimiter=',')
arr_labels = np.genfromtxt('./data/data_preparation/annotations/is_face_test_2.csv', delimiter=',')
arr_images = np.reshape(arr_images, (len(arr_images), 50, 50, 1))
# # Reshaping the input image
# x = arr_images[105]
# x = x.reshape((1,) + x.shape)

arr_images_test = np.copy(arr_images)
arr_labels_test = np.copy(arr_labels)

print(len(arr_images))
print(len(arr_labels))

for i in range(0, len(arr_images)):
    print("running image ", i)
    x = arr_images[i]
    label = arr_labels[i]
    x = x.reshape((1,) + x.shape)
    j = 0
    for batch in datagen.flow(x, batch_size=1):
        batch = np.squeeze(batch, axis=0)
        batch = np.reshape(batch, (-1))
        arr_images_test = np.append(arr_images_test, [batch])
        arr_labels_test = np.append(arr_labels_test, [label])
        j += 1
        print("\tbatch", j)
        if j > 2:
            break

arr_images_test = np.reshape(arr_images_test, (-1, 2500))
np.savetxt('./data/data_preparation/images/images_test_3.csv', arr_images_test, delimiter=",")
np.savetxt('./data/data_preparation/annotations/is_face_test_3.csv', arr_labels_test, delimiter=",")
