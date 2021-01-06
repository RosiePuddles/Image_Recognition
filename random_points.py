from setup import *
from time import time


class Image:
    def __init__(self, image, offset, size, scaled_size):
        self.image = image
        self.offset = offset
        self.size = size
        self.scaled_size = scaled_size

    def __repr__(self):
        return f'Dimensions in parent image : {self.size}' \
               f'Offset in parent image     : {self.offset}'


def linear_check(arr1: np.ndarray, arr2: np.ndarray, axis: int) -> np.ndarray:
    return np.equal(np.greater(arr1.sum(axis=axis), 0).astype(int) + np.greater(arr2.sum(axis=axis), 0).astype(int), 2)


def point_based() -> bool or Image:
    def check_params():
        if True not in pix_top_left: return False
        if True not in pix_top_right: return False
        if True not in pix_bottom_left: return False
        if True not in pix_bottom_right: return False
        return True

    def plot():
        fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        ax.imshow(pix_top_left)
        ax.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
        ax1.imshow(pix_top_right)
        ax1.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
        ax2.imshow(pix_bottom_left)
        ax2.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
        ax3.imshow(pix_bottom_right)
        ax3.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
        plt.show()

    def where_to_rows(arr: np.ndarray, height: int) -> np.ndarray:
        """
        Takes in an image and returns a 2D array where each item represents a row in the image and each
        value in the array is the index in the image where the pixel is True
        :param arr: array to be manipulated
        :param height: Height of the parent image
        :return: array of values that I can't be bothered to explain
        """
        a = np.array([*np.where(arr)]).transpose()
        b = np.zeros(height, dtype=object) - 1
        c = np.where(np.append(0, np.diff(a[:, 0])) > 0)
        b[np.take(a[:, 0], np.append(0, c))] = np.split(a[:, 1], *c)
        return b

    path = "./training/micro.jpg"
    # path = "./training/test1.jpg"
    parent_image, parent_dims, sub_image, offset, size = setup_image(path, retoffset=True, retsize=True)
    t0 = time()

    # Possible corner pixel locations
    corner_colors = np.array([sub_image[0, 0], sub_image[0, -1], sub_image[-1, 0], sub_image[-1, -1]])
    pix_top_left = np.equal(parent_image, corner_colors[0])
    pix_top_right = np.equal(parent_image, corner_colors[1])
    pix_bottom_left = np.equal(parent_image, corner_colors[2])
    pix_bottom_right = np.equal(parent_image, corner_colors[3])

    if not check_params(): return False

    for i in range(2):
        # Top Row
        pix_top_row_index = linear_check(pix_top_left, pix_top_right, 1)

        # Top Left
        pix_top_left = np.multiply(pix_top_left.transpose(), pix_top_row_index).transpose()
        pix_left_col_index = linear_check(pix_top_left, pix_bottom_left, 0)
        pix_top_left = np.multiply(pix_top_left, pix_left_col_index)

        # Top Right
        pix_top_right = np.multiply(pix_top_right.transpose(), pix_top_row_index).transpose()
        pix_right_col_index = linear_check(pix_top_right, pix_bottom_right, 0)
        pix_top_right = np.multiply(pix_top_right, pix_right_col_index)

        # Bottom Row
        pix_bottom_row_index = linear_check(pix_bottom_left, pix_bottom_right, 1)

        # Bottom Left
        pix_bottom_left = np.multiply(pix_bottom_left.transpose(), pix_bottom_row_index).transpose()
        pix_bottom_left = np.multiply(pix_bottom_left, pix_left_col_index)

        # Bottom Right
        pix_bottom_right = np.multiply(pix_bottom_right.transpose(), pix_bottom_row_index).transpose()
        pix_bottom_right = np.multiply(pix_bottom_right, pix_right_col_index)

    # # Filtering out values for bottom points above the possible highest top points
    # min_y = max(np.argmax(np.sum(pix_top_left, axis=1) > 0), np.argmax(np.sum(pix_top_right, axis=1) > 0))
    # min_y = np.zeros(parent_dims[0]) + np.append(np.zeros(min_y), np.ones(parent_dims[0] - min_y))
    # pix_top_left = np.multiply(pix_top_left.transpose(), min_y).transpose()
    # pix_top_right = np.multiply(pix_top_right.transpose(), min_y).transpose()
    # pix_bottom_left = np.multiply(pix_bottom_left.transpose(), min_y).transpose()
    # pix_bottom_right = np.multiply(pix_bottom_right.transpose(), min_y).transpose()
    #
    # # Filtering out values for top points below the possible lowest bottom points
    # max_y = parent_dims[0] - max(np.argmax(np.flip(np.sum(pix_bottom_left, axis=1)) > 0),
    #                              np.argmax(np.flip(np.sum(pix_bottom_right, axis=1)) > 0))
    # max_y = np.zeros(parent_dims[0]) + np.append(np.ones(max_y), np.zeros(parent_dims[0] - max_y))
    # pix_top_left = np.multiply(pix_top_left.transpose(), max_y).transpose()
    # pix_top_right = np.multiply(pix_top_right.transpose(), max_y).transpose()
    # pix_bottom_left = np.multiply(pix_bottom_left.transpose(), max_y).transpose()
    # pix_bottom_right = np.multiply(pix_bottom_right.transpose(), max_y).transpose()

    # top_left_all = np.array([*np.where(pix_top_left)]).transpose()
    # top_right_all = np.array([*np.where(pix_top_right)]).transpose()
    # bot_left_all = np.array([*np.where(pix_bottom_left)]).transpose()
    # bot_right_all = np.array([*np.where(pix_bottom_right)]).transpose()

    top_left_all = np.array([*np.where(pix_top_left)]).transpose()
    top_right_all = np.array([*np.where(pix_top_right)]).transpose()
    bot_left_all = np.array([*np.where(pix_bottom_left)]).transpose()
    bot_right_all = np.array([*np.where(pix_bottom_right)]).transpose()

    if not check_params(): return False

    # Find the row or column with the least possible number of combinations
    s1 = np.sum(pix_top_left)
    s2 = np.sum(pix_top_right)
    s3 = np.sum(pix_bottom_left)
    s4 = np.sum(pix_bottom_right)

    # top row, bottom row, left column, right column
    min_line = np.array([s1 * s2, s3 * s4, s1 * s3, s2 * s4]).argmin()

    if min_line == 0:
        pass
    elif min_line == 1:
        pass
    elif min_line == 2:
        pass
    else:
        pass

    print(f'{round(1000 * (time() - t0), 3)}ms elapsed')
    for i, n in [pix_top_left, 'top left'], [pix_top_right, 'top right'], \
                [pix_bottom_left, 'bottom left'], [pix_bottom_right, 'bottom right']:
        print(f'{round(100 * np.mean(i), 5)}% remaining for {n} pixel | {np.sum(i)} possible pixel(s)')

    plot()
    return True


if __name__ == "__main__":
    res = point_based()
    if res is False:
        print('Sub-image not located\nTry again with a larger threshold value.')
    else:
        print(res)
