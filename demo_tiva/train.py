import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, ReLU, Softmax 
from tensorflow.keras.models import Sequential

import os
MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape)

model = Sequential()

model.add(Conv2D(3, kernel_size=(4,4), input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(2, kernel_size=(3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(10))

model.add(Softmax())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5)


model.save(MODEL_TF)


