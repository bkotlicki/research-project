import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

from keras_preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1))


arr_images = np.genfromtxt('drive/MyDrive/RP/images_of_characters (2).csv', delimiter=',')
arr_labels = np.genfromtxt('drive/MyDrive/RP/annotations_of_characters (2).csv', delimiter=',')

arr_images = arr_images.reshape(arr_images.shape[0] // 7500, 50, 50, 3)

arr_images_test = np.copy(arr_images)
arr_labels_test = np.copy(arr_labels)

assert len(arr_images_test) == len(arr_labels_test)

# print(arr_images.shape)

# for i in range(0, len(arr_images)):
#     print("running image ", i)
#     x = arr_images[i]
#     label = arr_labels[i]
#     x = x.reshape((1,) + x.shape)

#     # print(x)

#     j = 0
#     for batch in datagen.flow(x, batch_size=1):
#         batch = np.squeeze(batch, axis=0)
#         batch = np.reshape(batch, (-1))
#         batch = np.reshape(batch, (50, -1))
#         batch = np.reshape(batch, (-1))
#         arr_images_test = np.append(arr_images_test, [batch])
#         arr_labels_test = np.append(arr_labels_test, [label])
#         # print(len(arr_images_test), len(arr_labels_test))
#         # assert len(arr_images_test) == len(arr_labels_test)
#         j += 1
#         # print("\tbatch", j)
#         if j > 6:
#             break




# assert len(data_images) == len(data_labels)
assert len(arr_images) == len(arr_labels)

# arr_images_test = arr_images_test.reshape(arr_images_test.shape[0] // 7500, 50, 50, 3)

assert len(arr_images_test) == len(arr_labels_test)

p = np.random.permutation(len(arr_images_test))
arr_images_test = arr_images_test[p]
arr_labels_test = arr_labels_test[p]

arr_labels_test = arr_labels_test - 1

arrs = np.array([])

for l in arr_labels_test:
  vec = np.zeros((8,))
  np.put(vec, [int(l)], [1])
  arrs = np.append(arrs, vec, axis=0)

arr_labels_test = []
arrs = arrs.reshape((len(arr_images_test), 8))
arr_labels_test = arrs

l = int(len(arr_images_test) * 0.7)
m = l + int(len(arr_images_test) * 0.15)
r = m + int(len(arr_images_test) * 0.15)


arr_images = arr_images_test / 255.

test_images = arr_images_test[l:m]
test_labels = arr_labels_test[l:m]

validate_images = arr_images_test[m:r]
validate_labels = arr_labels_test[m:r]

train_images = arr_images_test[:l]
train_labels = arr_labels_test[:l]

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 3)))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-3),
              metrics='accuracy')


history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(validate_images, validate_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# model.save('saved_model/my_model')