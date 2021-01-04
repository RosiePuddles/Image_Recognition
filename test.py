from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect
from functions import *
import sys
import cv2
import time

if __name__ == "__main__":
    #########
    # SETUP #
    #########

    current_module = sys.modules[__name__]
    fig, (ax, ax1) = plt.subplots(2)
    # Import parent image and convert to black and white
    parent_image = cv2.cvtColor(cv2.imread("./training/test1.jpg"), cv2.COLOR_RGB2GRAY)
    # Store parent image dimensions as array to reduce lookup times
    parent_dims = np.array([n for n in parent_image.shape])
    # Calculate the size, offset, and scaled size of the sub-image
    offset = np.random.randint(0, [int(n) for n in parent_dims / 2])
    print(offset)
    size = np.random.randint([int(n) for n in parent_dims / 5], [int(n) for n in parent_dims / 2])
    scale = np.random.randint(size, size * 2)
    # Crop the parent image using offset and size to create the sub image
    sub_image = parent_image[offset[0]:offset[0] + size[0]].transpose()[offset[1]:offset[1] + size[1]].transpose()
    # Resize the sub-image in the x direction
    sub_image = np.array([np.interp(np.linspace(0, 1, scale[1]), np.linspace(0, 1, size[1]), n) for n in sub_image])
    # Resize the sub-image in the y-direction
    sub_image = np.array([np.interp(np.linspace(0, 1, scale[0]), np.linspace(0, 1, size[0]), n) for n in
                          sub_image.transpose()]).transpose()
    # Plot the two images
    imgplot = ax.imshow(parent_image)
    imgplot = ax1.imshow(sub_image)
    # Add a rectangle onto the parent image to show the position of the sub-image
    ax.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
    # plt.show()

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
    t0 = time.time()
    for i in range(parent_dims[0]):
        F = derivatives(fsc(parent_image[i]))
        for n in t:
            if -0.8 < parent_image[i][np.int(n / dt)] - C0[0] < 0.8:
                P[0] += 1
                C = np.array([value(np.array([n]), F[0]), *[value_derivative(np.array([n]), m) for m in F[1:]]])
                R = C[1:] / C0[1:]
                R = np.power(R, np.array([3, 3 / 2, 1]))
                test = mse(R, np.tile(np.mean(R), number_of_derivatives))
                if test <= 0.001:
                    positions.append([i, int(n / dt), test])
                    P[1] += 1
    print(f'{round(time.time() - t0, 3)}s total for {parent_dims[0] * parent_dims[1]} pixels (average of '
          f'{round((time.time() - t0) / (parent_dims[0] * parent_dims[1]), 8)}s per pixel)\n{P[0]} '
          f'({round(100 * P[0] / (parent_dims[0] * parent_dims[1]), 6)}%) possible points found, {P[1]} '
          f'({round(100 * P[1] / (parent_dims[0] * parent_dims[1]), 6)}%) of which satisfied the requirements')
    [print(n) for n in positions]
    ax1.plot([n[0:2] for n in positions], 'k+')
    plt.show()
