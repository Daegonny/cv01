import cv2
import numpy as np
import pandas as pd
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


#normalizing
mi_1 = np.mean(frame_mean)
sigma_1 = np.std(frame_mean)

mi_2 = np.mean(frame_stddv)
sigma_2 = np.std(frame_stddv)

mi_3 = np.mean(frame_contrast)
sigma_3 = np.std(frame_contrast)


alpha_1 = (sigma_2/sigma_1)*mi_1 - mi_2
beta_1 = sigma_1/sigma_2

alpha_2 = (sigma_3/sigma_1)*mi_1 - mi_3
beta_2 = sigma_1/sigma_3

frame_stddv_new = [beta_1*(value + alpha_1) for value in frame_stddv]
frame_contrast_new = [beta_2*(value + alpha_2) for value in frame_contrast]

d = {"mean": frame_mean, "stddv": frame_stddv_new, "contrast": frame_contrast_new}
df = pd.DataFrame(d)
df.to_csv("data.csv", sep=';')

plt.plot(frame_mean, color='b')
plt.plot(frame_stddv_new, color='r')
plt.plot(frame_contrast_new, color='g')
plt.xlim([0,150])
plt.show()
