import numpy as np
import torch

f = np.ones([5, 5])
g = np.ones([3, 3])
h = np.zeros([3, 3])
h[0, 0] = f[0, 0] * g[0, 0] + f[0, 1] * g[0, 1] + f[0, 2] * g[0, 2] + f[1, 0] * g[1, 0] + f[1, 1] * g[1, 1] + f[1, 2] *\
          g[1, 2] + f[2, 0] * g[2, 0] + f[2, 1] * g[2, 1] + f[2, 2] * g[2, 2]
print(h[0, 0])


def conv(f1, g1):
    fx, fy = f1.shape
    gx, gy = g1.shape
    h_conv = np.zeros(fx-gx+1, fy-gy+1)
    hx, hy = h_conv.shape
    for x1 in range(hx):
        for y1 in range(hy):
            sum_1 = 0
            for x2 in range(gx):
                for y2 in range(gy):
                    sum_1 += f1[x1+x2, y1+y2] * g1[x2, y2]
            h[x1, y1] = sum_1

    return h


def padding(mat):
    x, y = mat.shape
    h_padding = np.zeros([x+2, y+2])
    for x1 in range(x):
        for y1 in range(y):
            h[x1+1, y1+1] = mat[x1, y1]

    return h


def conv_test():
    b1 = torch.eye(5)
    b2 = b1.view(1, 1, 5, 5)
    b3 = torch.ones(3, 3)
    b4 = b3.view(1, 1, 3, 3)
    b5 = torch.nn.functional.conv2d(b2, b4)
    print(b5)
    b6 = torch.nn.functional.conv2d(b2, b5, padding=1)
    print(b6)


if __name__ == '__main__':
    conv_test()