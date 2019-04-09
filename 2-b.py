import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('chaplin.mp4')

if not cap.isOpened():
    raise IOError("Could not open video")

def get_adjacent_pixels(img, x, y):
    h = img.shape[0]
    w = img.shape[1]
    pixels = []
    if x > 0: #it isn't at the first column
        pixels.append(img[y][x-1])
    if x < w-1: #it isn't at the last column
        pixels.append(img[y][x+1])
    if y > 0: #it isn't at the first row
        pixels.append(img[y-1][x])
    if y < h-1: #it isn't at the first row
        pixels.append(img[y+1][x])
    return pixels

def compute_contrast(image):
    h = image.shape[0]
    w = image.shape[1]
    contrast = 0
    for y in range(0, h):
        for x in range(0, w):
            contrast += (1/(h*w))*(np.abs(image[y,x]-np.mean(get_adjacent_pixels(image,x,y))))
    return contrast

frame_mean = []
frame_stddv = []
frame_contrast = []
count = 0;
while True:
    ret, frame = cap.read()

    if ret:
        count = count + 1;
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_stddv = cv2.meanStdDev(frame)
        m = float(mean_stddv[0][0])
        s = float(mean_stddv[1][0])
        c = compute_contrast(frame)
        frame_mean.append(m)
        frame_stddv.append(s)
        frame_contrast.append(c)
        print(count,(count/150)*100)
    else:
        break

cap.release()

plt.plot(frame_mean, color='b')
plt.plot(frame_stddv, color='r')
plt.plot(frame_contrast, color='g')
plt.xlim([0,150])
plt.show()
