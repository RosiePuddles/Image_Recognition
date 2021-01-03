import numpy as np
import cv2
import matplotlib.pyplot as plt

from functions import *

np.set_printoptions(threshold=10)


# img = cv2.cvtColor(cv2.imread("./training/1042-1200x600.jpg"), cv2.COLOR_RGB2GRAY)
# img2 = cv2.cvtColor(cv2.imread("./training/crop1_resize.jpg"), cv2.COLOR_RGB2GRAY)
#
# img_ffs = np.array([fsc(n, n.size, 2 * n.size) for n in img], dtype=np.ndarray)
# img2_ffs = np.array([fsc(n, n.size, 2 * n.size) for n in img2], dtype=np.ndarray)
#
# img_derivatives = np.array([img_ffs[0], derivatives(img_ffs[1]), derivatives(img_ffs[2], sin=True)], dtype=np.ndarray)
# img2_derivatives = np.array([img2_ffs[0], derivatives(img2_ffs[1]), derivatives(img2_ffs[2], sin=True)], dtype=np.ndarray)
# print(img_derivatives)
#
# a = V(0, img2_derivatives[0])
#
# for i in range(img_derivatives.shape[0]):
#     for n in range(img_derivatives.shape[1]):
#         # if mse()
#         pass

N = 600  # Number of samples

T = np.linspace(-np.pi, np.pi, N)

# x = np.sin(50*T)
# x = np.power(T, 2)
x = np.sin(T)

xf = fsc(x, 2 * np.pi)
n = xf[1].shape[0]

pre1 = np.linspace(0, 1, 100)
pre2 = xf[1].shape[0]
pre = np.zeros(100)
for i in pre1:
    pass
part1 = np.multiply(xf[1], np.cos(np.multiply(pre1.reshape((100, 1)), np.arange(1, pre2 + 1))))
part2 = np.multiply(xf[2], np.sin(np.multiply(pre1.reshape((100, 1)), np.arange(1, pre2 + 1))))
pre = np.sum(part1 + part2, axis=0)
# print(part1)
# print(part2)
plt.plot(pre)
plt.show()

# print(f'MSE: {mse(np.divide(1, np.power(np.arange(1, xf[1].shape[0] + 1), 2)), xf[1])}')
