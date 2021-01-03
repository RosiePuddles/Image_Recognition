from matplotlib import pyplot as plt
from functions import *
import sys
import cv2

if __name__ == "__main__":
    current_module = sys.modules[__name__]
    a = (0.7, np.array([1, 0.2, -0.7]), np.array([0, -0.9, -1.3]))
    a = derivatives(a)
    t = np.linspace(0, 2 * np.pi, 100)
    v = value(t, a[0])
    # fig, (val, d1, d2, d3, d4) = plt.subplots(1, 5)
    # val.plot(t, v)
    # [(getattr(current_module, f'd{n}').plot(t, value_derivative(t, a[n])), getattr(current_module, f'd{n}')
      # .set_title(f'Derivative {n}')) for n in np.arange(1, 5)]
    # plt.show()
