from functions import *
from setup import *
import sys
from time import time

if __name__ == "__main__":
    #########
    # SETUP #
    #########
    current_module = sys.modules[__name__]
    parent_image, parent_dims, sub_image = setup_image("./training/test1.jpg")

    ##############################
    # CALCULATION AND PROCESSING #
    ##############################
    # Calculate the C matrix for the top left corner of the
    C0 = derivatives(fsc(sub_image[0]))
    C0 = np.array([value(np.array([0]), C0[0]), *[value_derivative(np.array([0]), n) for n in C0[1:]]])
    # Create a time array for from 0 to 2 pi with equally spaced for each pixel in the parent image
    t, dt = np.linspace(0, 2 * np.pi, parent_dims[1], endpoint=False, retstep=True)
    # Store the number of derivatives for faster lookup times
    number_of_derivatives = C0.shape[0] - 1
    # Create empty arrays for recording the number of locations found for different requirements
    P = [0, 0]
    # Create an empty array for storing the possible positions of the top left pixel of the sub-image on parent image
    positions = []
    t0 = time()
    for i in range(parent_dims[0]):
        F = derivatives(fsc(parent_image[i]))
        for n in t:
            if -0.8 < parent_image[i][np.int(n / dt)] - C0[0] < 0.8:
                P[0] += 1
                C = np.array([value(np.array([n]), F[0]), *[value_derivative(np.array([n]), m) for m in F[1:]]])
                R = C[1:] / C0[1:]
                R = np.power(R, np.array([3, 3 / 2, 1]))
                test = mse(R, np.tile(np.mean(R), number_of_derivatives))
                if test <= 0.0001:
                    positions.append([int(n / dt), i, test])
                    P[1] += 1
    print(f'{round(time() - t0, 3)}s total for {parent_dims[0] * parent_dims[1]} pixels (average of '
          f'{round((time() - t0) / (parent_dims[0] * parent_dims[1]), 8)}s per pixel)\n{P[0]} '
          f'({round(100 * P[0] / (parent_dims[0] * parent_dims[1]), 6)}%) possible points found, {P[1]} '
          f'({round(100 * P[1] / (parent_dims[0] * parent_dims[1]), 6)}%) of which satisfied the requirements')
    [print(n) for n in positions]
