# C.V. Problem Set 01
Ígor Chagas Marques - Undergraduate Program - 201512040525
## Enviroment Details
Computer and enviroment specifications:
- Laptop Asus K47VM
- i7-3610QM CPU @ 2.30GHz 
- 8GB RAM
- Nvidia Geforce GT630M 2GB
- Elementary OS 0.4.1 Loki 64-bit (Built on Ubuntu 16.04.5 LTS)
- Python 2.7
- OpenCV 2.4.9.1.

## Problem 1
In this problem I selected this nice grasshopper as the interest image. It's a PNG image with three channels and 400 x 300px.

![Grasshopper](/home/daegonny/code/python/cv/ps01/grasshopper.png)

## 1.1 Loading and Displaying
Since all the files are placed at the same folder, the code below loads and displays the grasshopper!

#### 1-1.py
```python
import cv2

img = cv2.imread("grasshopper.png")
cv2.imshow("Grasshopper", img)
cv2.waitKey()
```

## 1.2 Histograms
After loading up the image, the histograms can be computed one by one and displayed in a single plot (with matplotlib helps) as it follows:

#### 1-2.py
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('grasshopper.png')
color = ('b','g','r')

for i,col in enumerate(color):
     histr = cv2.calcHist([img],[i],None,[256],[0,256])
     plt.plot(histr,color = col)
     plt.xlim([0,256])

plt.show()

```

![Histogram](/home/daegonny/code/python/cv/ps01/histogram.png)

## 1.3 Window_p
The below implementation uses mouse openCV callback to get its (x,y) coordinates whenever it moves. Then it draws the window_p rectangle shape as it moves. With the pixels of this window it computes mean, standard deviation and prints it with coordinates, RGB and intensity on the terminal window.

#### 1-3.py
```python
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
```
By example we have our window_p placed at grass on the (55,272) position. Thus the last terminal output gives us (61, 92, 2) for RGB value, 51 for intensity, (122.91, 150.74, 37.07) for mean and (39.01, 41.15, 44.12) for standard deviation:

![Window](/home/daegonny/code/python/cv/ps01/canvas.png)

## 1.4 Homogeneous and Inhomogeneous Areas
By definition the standard deviation is a measure that is used to quantify the amount of variation or dispersion of a set of data values. So we can use it to define an area as homogeneous (lower stard deviation) or inhomogeneous (higher standard deviation).

I've changed the previous code to compute values only when the mouse clicks instead of just moving around. Thus the new one looks like:

#### 1-4.py
```python
import cv2
import numpy as np

#define window_p size
window_half_size = 10
#define image
img = cv2.imread('grasshopper.png')
#backup original image
img_copy = img.copy()
img_window = img[0:0+2*window_half_size, 0:0+2*window_half_size]

def mouse_move_click(event, x, y, flags, param):
    #draw rectangle on mouse postion
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x-window_half_size, y-window_half_size), (x+window_half_size, y+window_half_size), (0,0,255))
        cv2.imshow('image', img_copy)

    if event == cv2.EVENT_LBUTTONUP:
        #cutting interest window
        img_window = img[y-window_half_size:y+window_half_size, x-window_half_size:x+window_half_size]
        cv2.rectangle(img, (x-window_half_size, y-window_half_size), (x+window_half_size, y+window_half_size), (255,0,0))
        cv2.imshow('image', img)
        #print info
        color = img[y,x]
        red = int(color[2])
        green = int(color[1])
        blue = int(color[0])
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
cv2.setMouseCallback("image", mouse_move_click)
cv2.waitKey(0)
```

So in order to evidentiate the difference between windows I placed the first again on the grass (385, 147) and the second one over the insect head (195, 116):

![Window-compare](/home/daegonny/code/python/cv/ps01/window_compare.png)

As we expected the grass window is more homogeneous, therefore it has lower standard deviation values.

## Problem 2
As an interest image sequence, I've chosen a short Chaplin video (150 frames and .mp4 format) with less than 7 seconds and one single channel.
The first frame can be checked below:

![Chaplin-frame](/home/daegonny/code/python/cv/ps01/chaplin-take.png)

## 2.a Reading Frames
During this course we had practical lessons at the labs with professor Nunes. In one of them he taught us to read a image sequence with the following code (available on https://github.com/cfgnunes/cv-lab/blob/master/08_reading_video.py):

#### 2-a.py
```python
import cv2

cap = cv2.VideoCapture('chaplin.mp4')

if not cap.isOpened():
    raise IOError("Could not open video")

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Input', frame)

    c = cv2.waitKey(10)
    if c == 27:
        break

cap.release()
```

At first place, it opens the file, and then with _read()_ function it reads a frame by time and displays it.

## 2.b Calculating Data Measures
Since these frames only have one channel, for this exercise I proposed the measures as follows: Mean, Standard Deviation and Contrast. The code computes these measures for each frame and appends it on a list. Then it plots it all together on the same axis:

#### 2-b.py
```python
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

```

![Mean-Stddv-Contrast-0](/home/daegonny/code/python/cv/ps01/mean_stddv_contrast_0.png)

Above we can see mean (blue), standard deviation (red) and contrast (green) values on each frame.

## 2.c Normalizing Measures
For a better comparison between different measures, it is ideal to map them onto functions with the same mean and variance. Assuming we have two functions **f(x)** and **g(x)** and we want to get **g_new(x)**. According to the course material, it can be done with the following formulas:

![Formula-1](/home/daegonny/code/python/cv/ps01/formula-1.png)
![Formula-2](/home/daegonny/code/python/cv/ps01/formula-2.png)

Where $\sigma_f$, $\sigma_g$ are its standard deviations and $\mu_f$,  $\mu_g$ are its means.

So normalizing and again plotting it all together on the same axis we  have:

#### 2-c.py
```python
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
```
![Mean-Stddv-Contrast-0](/home/daegonny/code/python/cv/ps01/mean_stddv_contrast_1.png)

As the previous exercise we can again see mean (blue), standard deviation (red) and contrast (green) values on each frame. But now these values are "normalized".

## 2.c L1 Metric

Working with Taxicab Geometry and the following formula, where _**p**_ and _**q**_ are function points and _**n**_ the set size:

![Taxicab](/home/daegonny/code/python/cv/ps01/taxicab.png)

We can compare the three functions two-by-two, using just the sum of the difference at the _**y**_ coordinate as our points are all aligned on the _**x**_ axis. Also we can divide the final values by the number of frames (150).

#### 2-d.py
```python
import numpy as np
import pandas as pd

df = pd.read_csv('data.csv', sep=';')
m_0 = 0
m_1 = 0
m_2 = 0


for index, row in df.iterrows():
    m_0 += np.abs(row['mean'] - row['stddv'])
    m_1 += np.abs(row['mean'] - row['contrast'])
    m_2 += np.abs(row['stddv'] - row['contrast'])

print("mean vs stddv: ", m_0/150)
print("mean vs contrast: ", m_1/150)
print("stddv vs contrast: ", m_2/150)

```
| Mean vs Stddv  | Mean vs Contrast | Stddv vs Contrast |
| -------------  | ---------------- | ----------------- |
| 2.250          | 10.593           | 10.006            |

As we can see mean and standard deviation are much more simmilar than  contrast mean and contrast or standard deviation and contrast. Maybe it's due the fact they era statistcly related.

## Problem 3
Here I've worked with different impacts of amplitudes and phase in frequency space on resulting filtered images, mostly using the FFT tool. 

## 3.a Frequency Domain Transform
I selected the images as two frames (492x359) of the previous exercise video:

#### image-1.png
![Image-1](/home/daegonny/code/python/cv/ps01/image-1.png)
#### image-2.png
![Image-2](/home/daegonny/code/python/cv/ps01/image-2.png)

Applying the DFT, getting magnitude and phase of both, making a new one and doing the inverse DFT we get the "combined" image.

#### 3-a.py
```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

image_1 = cv2.imread("image-1.png",0)
image_2 = cv2.imread("image-2.png",0)

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
cv2.imwrite("chaplin-combined.png", new_image)

```

#### chaplin-combined.png
![Chaplin-combined](/home/daegonny/code/python/cv/ps01/chaplin-combined.png)

Looking at this "combined" image we could infere that the phase is the winner.

## 3.b Magnitude and Angle Changes

I've selected two texture images to perform uniform changes on its magnitudes and angles. One of them is a leather surface and the other one is a wall of bricks:

#### leather-texture.png
![Leather-texture](/home/daegonny/code/python/cv/ps01/leather-texture.png)

#### brick-texture.png
![Brick-texture](/home/daegonny/code/python/cv/ps01/brick-texture.png)

Though here they have three channels, I had worked with them in a single channle (gray scale). All the four transformations were computed as a scalar multiplication by the factor 2 with the respective matrix (magnitude or angle).

#### 3-b.py
```python
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

```

### Leather Changes

#### new_mag_leather.png
![New-mag-leather](/home/daegonny/code/python/cv/ps01/new_mag_leather.png)

#### new_ang_leather.png
![New-ang-leather](/home/daegonny/code/python/cv/ps01/new_ang_leather.png)

### Brick Changes

#### new_mag_brick.png
![New-mag-brick](/home/daegonny/code/python/cv/ps01/new_mag_brick.png)

#### new_ang_brick.png
![New-ang-brick](/home/daegonny/code/python/cv/ps01/new_ang_brick.png)

As we can see the magnitude modification causes more significant changes in the image.

## 3.c Human Being Faces

In this exercise I've applied the same changes as I did in the previous but using a human being face as image. In this case my own face:

#### me.png
![Me](/home/daegonny/code/python/cv/ps01/me.png)

#### 3-c.py
```python
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

```

#### new_mag_me.png
![New-mag-me](/home/daegonny/code/python/cv/ps01/new_mag_me.png)

#### new_ang_me.png
![New-ang-me](/home/daegonny/code/python/cv/ps01/new_ang_me.png)

Unlike the texture image ones,the face got a big modification when I applied a angle change.

## Problem 4

To approximate the HSI Space we can "slice" a RGB cube with a plane orthogonal to the grey level vector. This equation could be denoted as: **X+Y+Z-3u = 0**. Where **_x_**, **_y_**, **_z_** are the axis of a 3D space and **_u_** stands for the cut position (it varies from 0 to 255).

The **_u_** value starts at **131** and it can be increased by one unit **clicking** on the image canvas or decreased by one unit holding **crtl** key and also **clicking** on the image canvas. Pressing any keyboard key will end the program.

**It is important to say that the process to calculate and plotting the new surface may take a little and it requires the user to wait it load up before making any changes.**

To calculate each surface we compute the intersection between the cube integer points with the surface, then we project them on a 2D space, then normalize it and for last but not least we plot them on the image canvas.

#### 4.py
```python
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

u = 131
satur = np.zeros((480,640))

def minus_u(u):
    if u > 0:
        return u - 1
    else:
        return u

def plus_u(u):
    if u < 255:
        return u + 1
    else:
        return u

def get_vector_plane(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    u = b-a
    v = b-c
    u_len = np.sqrt(np.power(u[0], 2) + np.power(u[1], 2) + np.power(u[2], 2))
    v_len = np.sqrt(np.power(v[0], 2) + np.power(v[1], 2) + np.power(v[2], 2))
    u = u/u_len
    v = v/v_len
    w = np.cross(u,v)
    u = np.cross(v,w)
    return u,v

def project_on_plane(p, u, v):
    y = np.dot(u,p)
    x = np.dot(v,p)
    return x,y

def is_point_on_plane(x,y,z, u):
    return (x + y + z - (3*u)) == 0

def is_collinear(a,b,c):
    ab = np.sqrt(np.power(a[0]-b[0], 2) + np.power(a[1]-b[1], 2) + np.power(a[2]-b[2], 2))
    ac = np.sqrt(np.power(a[0]-c[0], 2) + np.power(a[1]-c[1], 2) + np.power(a[2]-c[2], 2))
    bc = np.sqrt(np.power(b[0]-c[0], 2) + np.power(b[1]-c[1], 2) + np.power(b[2]-c[2], 2))
    m = np.max([ab, ac, bc])
    n = np.median([ab, ac, bc])
    o = np.min([ab, ac, bc])
    return not m < n + o

def get_non_coll(points):
    l = len(points)
    non_points = []
    if l >= 3:
         for i in range(l):
             if l - i < 3:
                 break
             if not is_collinear(points[i], points[i+1], points[i+2]):
                 return [points[i], points[i+1], points[i+2]]
    else:
        return non_points

def plane_points(u):
    rgb = []
    for x in range(256):
        for y in range(256):
            for z in range(256):
                if is_point_on_plane(x, y, z, u):
                    rgb.append([x,y,z])
    return rgb


def mouse_callback(event, x, y, flags, param):
    global u
    global img2
    if event == cv2.EVENT_LBUTTONUP:
        if flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON):
            u = minus_u(u)
            print("u = " + str(u))
            cut_cube(u)

        else:
            u = plus_u(u)
            print("u = " + str(u))
            cut_cube(u)

def get_satur(img):
    img = np.float64(img)
    satur = np.zeros((480,640))
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    for i in range(480):
        for j in range(640):
            if (r[i,j]+g[i,j]+b[i,j]) == 0:
                satur[i,j] = 0
            else:
                satur[i,j] = 1-3*((np.min([r[i,j],g[i,j],b[i,j]]))/(r[i,j]+g[i,j]+b[i,j]))
            # satur = np.min([r[i,j],g[i,j],b[i,j]])
    return satur


def cut_cube(u):
    print("Please wait! It may take a little.")
    pp = plane_points(u)
    nc = get_non_coll(pp)
    w,v = get_vector_plane(nc[0], nc[1], nc[2])

    xs = []
    ys = []
    hexes = []

    #projection
    for p in pp:
        x,y=project_on_plane(p, w, v)
        xs.append(x)
        ys.append(y)
        hexes.append('#{:02x}{:02x}{:02x}'.format(p[0], p[1] ,p[2]))

    #normalizing
    mi_1 = np.mean(ys)
    sigma_1 = np.std(ys)
    mi_2 = np.mean(xs)
    sigma_2 = np.std(xs)
    alpha_1 = (sigma_2/sigma_1)*mi_1 - mi_2
    beta_1 = sigma_1/sigma_2
    xs_new = [beta_1*(value + alpha_1) for value in xs]

    fig = plt.figure()
    plt.scatter(xs_new, ys, c=hexes)
    plt.axis('off')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    satur = get_satur(data)
    cv2.imshow("Cut", data)
    cv2.imshow("Saturation", satur)

cut_cube(u)
cv2.setMouseCallback("Cut", mouse_callback)
cv2.waitKey(0)

```

Below we can see the color cuts and its saturation values for **__u = 131__**, **_u = 150_** and **_u = 74_**, respectively: 

#### u-131.png
![u-131](/home/daegonny/code/python/cv/ps01/u-131.png)

#### u-150.png
![u-150](/home/daegonny/code/python/cv/ps01/u-150.png)

#### u-74.png
![u-74](/home/daegonny/code/python/cv/ps01/u-74.png)