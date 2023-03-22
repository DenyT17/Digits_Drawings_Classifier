import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Data normalization
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Data Augumentation
m=60000
datagen = ImageDataGenerator(
    width_shift_range= 2.0,
    height_shift_range= 2.0,
    rotation_range = 20,
)

datagen.fit(X_train.reshape(X_train.shape[0], 28, 28, 1))

data_generator = datagen.flow(X_train.reshape(X_train.shape[0], 28, 28, 1),shuffle=False, batch_size=1)

type(data_generator)

X_train_aug = [data_generator.next() for i in range(0, m * 4)]

X_train_240k = np.asarray(X_train_aug).reshape(m * 4, 28 * 28)

y_train_240k = np.concatenate((y_train, y_train, y_train, y_train))

# Building model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


# Compile and train model
epochs = 15
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train_240k, y_train_240k,validation_data=(X_test, y_test), epochs=epochs)


# Training result visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Saving the model
model.save('models/MNIST_MODEL_24k.h5')