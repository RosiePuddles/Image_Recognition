import numpy as np


def fsc(f) -> (np.float, np.ndarray, np.ndarray):
    """
    Given a periodic, function f(t) with period 2 pi, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    :param f : array of values to for which the coefficients wish to be calculated

    :returns a0 : constant offset of the periodic function
    :returns a: coefficients for cosine functions
    :returns b: coefficients for sine functions
    """
    length = f.shape[0] + 2
    y = np.divide(np.multiply(np.fft.rfft(f), 2), length)
    return y[0].real / 2, y[1:-1].real, -y[1:-1].imag


def mse(x: np.array, xc: np.array) -> np.float:
    """
    Takes in two arrays of the same size, x and xc, and calculates the mean squared error for the two arrays
    :param x: expected values
    :param xc: calculated values
    :return: float
    """
    if x.shape[0] != xc.shape[0]:
        return False
    else:
        return np.sum(np.square(x - xc))/x.shape[0]


def derivatives(x: tuple) -> np.ndarray:
    s = np.arange(1, x[1].shape[0] + 1)
    x_ = x[1:]
    n = np.array([[-1], [1]])
    d1 = np.multiply(np.multiply(s, x_), n)
    d2 = np.multiply(np.multiply(-s, d1), n)
    d3 = np.multiply(np.multiply(s, d2), n)
    # d4 = np.multiply(np.multiply(-s, d3), n)
    return np.array([x, np.array([d1[1], d1[0]]), d2, np.array([d3[1], d3[0]])], dtype=object)


def value(t: np.ndarray, c: tuple) -> np.ndarray:
    t = t.reshape(t.shape[0], 1)
    cos = np.multiply(c[1], np.cos(np.multiply(t, np.arange(1, c[1].shape[0] + 1))))
    sin = np.multiply(c[2], np.sin(np.multiply(t, np.arange(1, c[1].shape[0] + 1))))
    return np.sum(np.add(sin, cos), axis=1) + np.tile(c[0], t.shape[0])


def value_derivative(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    t = t.reshape(t.shape[0], 1)
    cos = np.multiply(c[0], np.cos(np.multiply(t, np.arange(1, c[1].shape[0] + 1))))
    sin = np.multiply(c[1], np.sin(np.multiply(t, np.arange(1, c[1].shape[0] + 1))))
    return np.sum(np.add(sin, cos), axis=1)
