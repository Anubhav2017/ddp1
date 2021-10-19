import serial
import time
import numpy as np
import os 
import cv2

image_dir="test_images"
listimage=os.listdir(image_dir)
runTotal = len(listimage)
classes = ['zero','one','two','three','four','five','six','seven','eight','nine'] 

imglist = []
truths = []
for i in range(runTotal):

    image_path = os.path.join(image_dir,listimage[i])
    # print listimage[i].split('_',2)
    a,ground_truth,_ = listimage[i].split('_',2)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.reshape(-1,)
    imglist.append(image)
    truths.append(ground_truth)

board = serial.Serial()
board.baudrate = 115200
board.port = '/dev/ttyACM0'    # Serial COM port (115200)
board.open()

imgindx = np.random.randint(0,10000,30)

imgs=[]
ys=[]


for i in range(30):
    imgs.append(imglist[imgindx[i]])
    ys.append(truths[imgindx[i]])
f=0
while(f==0):
    s = board.readline().strip()
    if(s=="start!"):
        f=1
    else:
        print(s)


i=0
while True:
    x=raw_input("Press enter to continue")

    for j in range(784):
        board.write(chr(imgs[i][j]))
    time.sleep(0.1)

    ans = board.readline()
    print(ans)
    print("actual label = {}".format(ys[i]))
    i+=1
    if i==30:
        i=0




# import serial
# import time
# from tensorflow.keras.datasets import mnist
# import numpy as np
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape)
# board=serial.Serial(port='COM10', baudrate=115200)

# imgindx = np.random.randint(0,9,10)
# imgs = X_test[imgindx]
# ys= y_test[imgindx]
# f=0
# while(f==0):
#     s = board.readline().strip().decode()
#     if(s=="start!"):
#         f=1
#     else:
#         print(s)

# imgs = np.reshape(imgs,(10,-1,))
# print(imgs.shape)
# i=0
# while True:
#     input("Press enter to load image")

#     # for j in range(784):
#     #     board.write(chr(imgs[i][j]).encode())
#     # board.write(imgs[i])
#     #time.sleep(0.1)

#     ans = board.readline().decode()
#     print(ans)
#     print("predicted output= {}".format(ans))
#     # print("actual label = {}".format(ys[i]))
#     i+=1
#     if i==10:
#         i=0