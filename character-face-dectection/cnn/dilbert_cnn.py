import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# (a, b), (c, d) = datasets.cifar10.load_data()
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images, test_images = train_images / 255.0, test_images / 255.0

# print(train_images[0])

data_images = np.genfromtxt('drive/MyDrive/RP/images_dilbert_3.csv', delimiter=',')
data_labels = np.genfromtxt('drive/MyDrive/RP/faces_dilbert_3.csv', delimiter=',')

data_images = data_images.reshape(data_images.shape[0] // 7500, 50, 50, 3)

# shuffle
assert len(data_images) == len(data_labels)
p = np.random.permutation(len(data_images))
data_images = data_images[p]
data_labels = data_labels[p]

# for i in range(2000, 2100):
#   print(data_labels[i])
#   cv2_imshow(data_images[i])

# splits
l = 4000
m = 4450
r = 4900

data_images = data_images / 255.

test_images = data_images[l:m]
test_labels = data_labels[l:m]

validate_images = data_images[m:r]
validate_labels = data_labels[m:r]

train_images = data_images[:l]
train_labels = data_labels[:l]

# train_images = train_images / 255.
# test_images = test_images / 255.
# validate_images = validate_images / 255.

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              metrics='accuracy')


history = model.fit(train_images, train_labels, epochs=15,
                    validation_data=(validate_images, validate_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

model.save('saved_model/my_model')