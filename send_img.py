import serial
import time
from tensorflow.keras.datasets import mnist
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
board=serial.Serial(port='COM10', baudrate=115200)

imgindx = np.random.randint(0,9,10)
imgs = X_test[imgindx]
ys= y_test[imgindx]
f=0
while(f==0):
    s = board.readline().strip().decode()
    if(s=="start!"):
        f=1
    else:
        print(s)

imgs = np.reshape(imgs,(10,-1,))
print(imgs.shape)
i=0
while True:
    input("Press enter to load image")

    # for j in range(784):
    #     board.write(chr(imgs[i][j]).encode())
    # board.write(imgs[i])
    #time.sleep(0.1)

    ans = board.readline().decode()
    print(ans)
    print("predicted output= {}".format(ans))
    # print("actual label = {}".format(ys[i]))
    i+=1
    if i==10:
        i=0