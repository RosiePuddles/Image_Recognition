import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from scipy.optimize import curve_fit


def width_color_range_correl(n):
    def random_positions(image) -> [float, int]:
        row = randint(0, image.shape[0] - 1)
        image = image[row]
        assert isinstance(image, np.ndarray)
        offset = randint(0, image.shape[0] - 2)
        offset = [offset, randint(offset + 1, image.shape[0])]
        image = image[offset[0]:offset[1]]
        col_range = np.max(image) - np.min(image)
        return [col_range, offset[1] - offset[0] + 1]

    for i in [f'./training/test{n}.jpg' for n in range(1, 4)]:
        parent_image = cv2.cvtColor(cv2.imread(i), cv2.COLOR_RGB2GRAY)
        plot_data = np.array([random_positions(parent_image) for _ in range(n)])
        plot_data = plot_data.transpose()
        plt.plot(*plot_data, '+')

    plt.xlabel('Color Range')
    plt.ylabel('Width of region')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()


def width_error_approximation(iterations, min_length, max_length, d=1):
    def mse(arr1, arr2):
        return np.sum(np.square(arr2 - arr1)) / arr1.shape[0]

    def fitted_func(x, a, b):
        return b + a/x

    max_values = np.zeros((max_length - min_length + 1, 2))
    min_values = np.zeros((max_length - min_length + 1, 2))
    mean_values = np.zeros((max_length - min_length + 1, 2))
    params = np.ones(2)

    for i in range(min_length + d, max_length + 1 + d):
        temp = np.array([])
        fit = np.random.rand(i)
        for _ in range(iterations):
            long = np.random.rand(i)
            short = long[:-d]
            delta = abs(mse(fit, long) - mse(fit, np.interp(np.linspace(0, 1, i), np.linspace(0, 1, i - d), short)))
            temp = np.append(temp, delta)

        mean_values[i - min_length - d] = [i - d, np.mean(temp)]
        max_values[i - min_length - d] = [i - d, np.max(temp)]
        min_values[i - min_length - d] = [i - d, np.min(temp)]

    np.savetxt('width_mean.csv', np.array([mean_values[:, 0], mean_values[:, 1]]).transpose(), delimiter=',')
    np.savetxt('width_max.csv', np.array([max_values[:, 0], max_values[:, 1]]).transpose(), delimiter=',')
    np.savetxt('width_min.csv', np.array([min_values[:, 0], min_values[:, 1]]).transpose(), delimiter=',')

    fitted = curve_fit(fitted_func, *(mean_values.transpose()), params)
    fitted = fitted_func(np.arange(min_length, max_length + 1), *fitted[0])
    print(fitted)

    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    ax.plot(*(max_values.transpose()), '+')
    ax.plot(np.arange(min_length, max_length + 1), fitted)
    ax.plot(*(mean_values.transpose()), '+')
    ax.plot(*(min_values.transpose()), '+')
    ax1.plot(*(max_values.transpose()), '+')
    ax2.plot(*(mean_values.transpose()), '+')
    ax2.plot(np.arange(min_length, max_length + 1), fitted)
    ax3.plot(*(min_values.transpose()), '+')
    plt.show()


width_error_approximation(2500, 10, 2500)
