import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  Sequential
from keras.models import load_model
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import matplotlib.pyplot as plt
import cv2

batch_size = 32
img_size = 	256

data = tf.keras.utils.image_dataset_from_directory(r'images')
class_names=data.class_names
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

data = data.map(lambda x,y: (x/255,y))


train_size = int(len(data)*0.7)+1
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


model = Sequential()
model.add(Conv2D(32,(3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train,epochs=5,validation_data=val)

def plot_hist(hist):
    plt.figure(figsize=(5, 5))
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.figure(figsize=(5, 5))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(history)

model.save('Model10kimg.h5')
model = load_model('models/Model10kimg.h5')
test_evaluate=model.evaluate(test)
print('Accuracy classification for test data = {}, and loss = {}'
      .format(test_evaluate[1],test_evaluate[0]))