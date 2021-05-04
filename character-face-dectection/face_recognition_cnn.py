# to be ran on Google Colab
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

train_images = np.genfromtxt('drive/MyDrive/RP/images_test.csv', delimiter=',')
train_labels = np.genfromtxt('drive/MyDrive/RP/is_face_test.csv', delimiter=',')

test_images = np.genfromtxt('drive/MyDrive/RP/images_test.csv', delimiter=',')
test_labels = np.genfromtxt('drive/MyDrive/RP/is_face_test.csv', delimiter=',')



train_images = train_images[:3000]
train_labels = train_labels[:3000]

test_images = test_images[3000:3360]
test_labels = test_labels[3000:3360]

train_images = train_images / 255.
test_images = test_images / 255.

# # reshape the array of train images and test images
train_images = np.reshape(train_images, (3000, 50, 50, 1))
test_images = np.reshape(test_images, (360, 50, 50, 1))

# for i in range(2000, 2350):
#   print(train_labels[i])
#   cv2_imshow(train_images[i])

# reshape the array of train labels and test labels
train_labels = np.reshape(train_labels, (3000, 1))
test_labels = np.reshape(test_labels, (360, 1))

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics='accuracy')


history = model.fit(train_images, train_labels, epochs=15,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# model.save('saved_model/my_model')
