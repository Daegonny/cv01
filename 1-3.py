import cv2
import numpy as np

#define window_p size
window_half_size = 10
#define image
img = cv2.imread('grasshopper.png')
#backup original image
img_copy = img.copy()
img_window = img[0:0+2*window_half_size, 0:0+2*window_half_size]

def mouse_move(event, x, y, flags, param):
    #draw rectangle on mouse postion
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x-window_half_size, y-window_half_size), (x+window_half_size, y+window_half_size), (0,0,255))
        cv2.imshow('image', img_copy)

        #cutting interest window
        img_window = img[y-window_half_size:y+window_half_size, x-window_half_size:x+window_half_size]
        #print info
        color = img[y,x]
        red = color[2]
        green = color[1]
        blue = color[0]
        intensity = (red+green+blue)/3
        mean_stddv = cv2.meanStdDev(img_window)
        print("========Point and Window Info========")
        print("Position",(x,y))
        print("(R,G,B)",(red, green, blue))
        print("Intensity", intensity)
        print("Mean",(float(mean_stddv[0][2]), float(mean_stddv[0][1]), float(mean_stddv[0][0])))
        print("Stddv",(float(mean_stddv[1][2]), float(mean_stddv[1][1]), float(mean_stddv[1][0])))
        print("=====================================")

cv2.imshow('image', img)
#set callback
cv2.setMouseCallback("image", mouse_move)
cv2.waitKey(0)
