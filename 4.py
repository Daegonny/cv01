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
        return u - 19
    else:
        return u

def plus_u(u):
    if u < 255:
        return u + 19
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
    print("Finished!")

cut_cube(u)
cv2.setMouseCallback("Cut", mouse_callback)
cv2.waitKey(0)
