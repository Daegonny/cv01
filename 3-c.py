import numpy as np
import cv2

me = cv2.imread("me.png",0)

new_mag_me = np.zeros([320, 320, 2])
new_ang_me = np.zeros([320, 320, 2])

cv2.imshow("me Texture", me)

dft_me = cv2.dft(np.float64(me),flags = cv2.DFT_COMPLEX_OUTPUT)
mag_me, ang_me = cv2.cartToPolar(dft_me[:,:,0], dft_me[:,:,1])

real_mag_me, im_mag_me = cv2.polarToCart(2*mag_me, ang_me)
real_ang_me, im_ang_me = cv2.polarToCart(mag_me, 2*ang_me)

new_mag_me[:,:,0], new_mag_me[:,:,1] = real_mag_me, im_mag_me
new_ang_me[:,:,0], new_ang_me[:,:,1] = real_ang_me, im_ang_me


new_mag_me = cv2.idft(new_mag_me, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
new_ang_me = cv2.idft(new_ang_me, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

cv2.imshow("Me Mag Change", np.array(new_mag_me, dtype = np.uint8))
cv2.imshow("Me Ang Change", np.array(new_ang_me, dtype = np.uint8))

cv2.imwrite("new_mag_me.png", np.array(new_mag_me, dtype = np.uint8))
cv2.imwrite("new_ang_me.png", np.array(new_ang_me, dtype = np.uint8))

cv2.waitKey()
