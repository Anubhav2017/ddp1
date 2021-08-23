import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range(5):
    imgflat = np.reshape(X_train[i],(784))
    print("img{} = {}".format(y_train[i],repr(imgflat)))