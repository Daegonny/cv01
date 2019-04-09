import numpy as np
import cv2
from matplotlib import pyplot as plt

image_1 = cv2.imread("image-1.png",0)
image_2 = cv2.imread("image-2.png",0)
cv2.imshow("image 1", image_1)
cv2.imshow("image 2", image_2)

image_new = np.zeros([359, 492, 2])

dft_1 = cv2.dft(np.float64(image_1),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_2 = cv2.dft(np.float64(image_2),flags = cv2.DFT_COMPLEX_OUTPUT)

mag_1, ang_1 = cv2.cartToPolar(dft_1[:,:,0], dft_1[:,:,1])
mag_2, ang_2 = cv2.cartToPolar(dft_2[:,:,0], dft_2[:,:,1])

x , y = cv2.polarToCart(mag_1, ang_2)

image_new[:,:,0] = x
image_new[:,:,1] = y
new_image = cv2.idft(image_new, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
cv2.imshow("combined", np.array(new_image, dtype = np.uint8))
cv2.waitKey()
cv2.imwrite("chaplin-combined.png", np.array(new_image, dtype = np.uint8))
