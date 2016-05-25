import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('kepka1.jpg')
image1 = np.array(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
image2 = cv2.imread('kepka2.jpg')
image2 = np.array(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
image3 = cv2.imread('kepka3.jpg')
image3 = np.array(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
images = (image1,image2,image3)
np.save("images",images)

gray1 = cv2.imread('mask1.bmp', 0)
gray2 = cv2.imread('mask2.bmp', 0)
gray3 = cv2.imread('mask3.bmp', 0)
_, binarizated1 = cv2.threshold(gray1,222,255,cv2.THRESH_BINARY_INV)
_, binarizated2 = cv2.threshold(gray2,222,255,cv2.THRESH_BINARY_INV)
_, binarizated3 = cv2.threshold(gray3,222,255,cv2.THRESH_BINARY_INV)
masks = (((np.array(binarizated1>0)),(np.array(binarizated2>0)),(np.array(binarizated3>0))))
np.save("seeds",masks)

check = np.load("images.npy")
plt.imshow(check[2])
plt.show()