import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect


def setup_image(path: str, plot: bool = False, retoffset: bool = False, retsize: bool = False, retscale: bool = False)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    # Import parent image and convert to black and white
    parent_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
    # Store parent image dimensions as array to reduce lookup times
    parent_dims = np.array([n for n in parent_image.shape])
    # Calculate the size, offset, and scaled size of the sub-image
    offset = np.random.randint(0, [int(n) for n in parent_dims / 2])
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
    if plot:
        fig, (ax, ax1) = plt.subplots(2)
        imgplot = ax.imshow(parent_image)
        imgplot = ax1.imshow(sub_image)
        # Add a rectangle onto the parent image to show the position of the sub-image
        ax.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
        plt.show()
    out = [parent_image, parent_dims, sub_image]
    out.append(offset) if retoffset else None
    out.append(size - 1) if retsize else None
    out.append(scale - 1) if retscale else None
    return out
