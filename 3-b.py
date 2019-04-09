import numpy as np
import cv2

leather = cv2.imread("leather-texture.png",0)
brick = cv2.imread("brick-texture.png",0)

new_mag_leather = np.zeros([320, 500, 2])
new_ang_leather = np.zeros([320, 500, 2])

new_mag_brick = np.zeros([183, 275, 2])
new_ang_brick = np.zeros([183, 275, 2])

cv2.imshow("Leather Texture", leather)
cv2.imshow("Brick Texture", brick)

dft_leather = cv2.dft(np.float64(leather),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_brick = cv2.dft(np.float64(brick),flags = cv2.DFT_COMPLEX_OUTPUT)

mag_leather, ang_leather = cv2.cartToPolar(dft_leather[:,:,0], dft_leather[:,:,1])
mag_brick, ang_brick = cv2.cartToPolar(dft_brick[:,:,0], dft_brick[:,:,1])

real_mag_leather, im_mag_leather = cv2.polarToCart(2*mag_leather, ang_leather)
real_ang_leather, im_ang_leather = cv2.polarToCart(mag_leather, 2*ang_leather)

real_mag_brick, im_mag_brick = cv2.polarToCart(2*mag_brick, ang_brick)
real_ang_brick, im_ang_brick = cv2.polarToCart(mag_brick, 2*ang_brick)

new_mag_leather[:,:,0], new_mag_leather[:,:,1] = real_mag_leather, im_mag_leather
new_ang_leather[:,:,0], new_ang_leather[:,:,1] = real_ang_leather, im_ang_leather

new_mag_brick[:,:,0], new_mag_brick[:,:,1] = real_mag_brick, im_mag_brick
new_ang_brick[:,:,0], new_ang_brick[:,:,1] = real_ang_brick, im_ang_brick

new_mag_leather = cv2.idft(new_mag_leather, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
new_ang_leather = cv2.idft(new_ang_leather, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

new_mag_brick = cv2.idft(new_mag_brick, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
new_ang_brick = cv2.idft(new_ang_brick, flags = cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

cv2.imshow("Leather Mag Change", np.array(new_mag_leather, dtype = np.uint8))
cv2.imshow("Leather Ang Change", np.array(new_ang_leather, dtype = np.uint8))

cv2.imshow("Brick Mag Change", np.array(new_mag_brick, dtype = np.uint8))
cv2.imshow("Brick Ang Change", np.array(new_ang_brick, dtype = np.uint8))

cv2.imwrite("new_mag_leather.png", np.array(new_mag_leather, dtype = np.uint8))
cv2.imwrite("new_ang_leather.png", np.array(new_ang_leather, dtype = np.uint8))

cv2.imwrite("new_mag_brick.png", np.array(new_mag_brick, dtype = np.uint8))
cv2.imwrite("new_ang_brick.png", np.array(new_ang_brick, dtype = np.uint8))

cv2.waitKey()
