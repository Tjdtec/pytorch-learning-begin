import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from mpl_toolkits.mplot3d import Axes3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x = np.linspace(-10, 10, 1000)
y = x ** 2
plt.plot(x, y)
plt.title("2-d function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X) + np.cos(Y) + 1
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
plt.show()


def grad(y, dy, x, eta, count=1000):
    dir = np.sign(y(x) - y(x - eta))
    ymin = y(x)
    xmin = x
    for epoch in range(count):
        xmin = x
        x = x - dir * eta * dy(x)
        yn = y(x)
        if yn < ymin:
            ymin = yn
        else:
            print("ymin=", ymin, "xmin=", xmin)
            print('yn=', yn)
            break
            pass
        print(ymin)
    return xmin, ymin


def grad3(x, y, z, dx, dy, eta, count=1000):
    dir_x = np.sign(z(x, y) - z(x - eta, y))
    dir_y = np.sign(z(x, y) - z(x, y - eta))
    zmin = z(x, y)
    xmin, ymin = x, y
    print(zmin)
    for i in range(count):
        xmin, ymin = x, y
        x = x - eta * dir * dx(x)
        y = y - eta * dir * dy(x)
        zn = z(x, y)
        if zn < z(x, y):
            zmin = zn
        else:
            print('zmin=', zmin, "x=", xmin, "y=", ymin)
            print('zn=', zn)
            break
        print(zmin)


if __name__ == '__main__':
    grad()
