from setup import *
from functions import mse
from time import time


class Result:
    def __init__(self, offset, size, time_taken, iterative_points):
        self.offset = offset
        self.size = size
        self.time_taken = 1000 * time_taken
        self.iterative_points = iterative_points

    def __repr__(self):
        col = '\033[31m' if self.time_taken > 25 else '\033[32m'
        return f'{"Time taken to find (ms)".ljust(28)} : {col}{round(self.time_taken, 3)}\033[0m\n' \
               f'{"Offset in parent image".ljust(28)} : \033[35m{self.offset}\033[0m\n' \
               f'{"Dimensions in parent image".ljust(28)} : \033[35m{self.size}\033[0m\n' \
               f'Iterative positions\n' \
               f'   {"Top Left".ljust(25)} : \033[36m{self.iterative_points[0]}\033[0m\n' \
               f'   {"Top Right".ljust(25)} : \033[36m{self.iterative_points[1]}\033[0m\n' \
               f'   {"Bottom Left".ljust(25)} : \033[36m{self.iterative_points[2]}\033[0m\n' \
               f'   {"Bottom Right".ljust(25)} : \033[36m{self.iterative_points[3]}\033[0m\n'


def linear_check(arr1: np.ndarray, arr2: np.ndarray, axis: int) -> np.ndarray:
    return np.equal(np.greater(arr1.sum(axis=axis), 0).astype(int) + np.greater(arr2.sum(axis=axis), 0).astype(int), 2)


def point_based() -> bool or Result:
    def check_params() -> bool:
        if True not in pix_top_left: return False
        if True not in pix_top_right: return False
        if True not in pix_bottom_left: return False
        if True not in pix_bottom_right: return False
        return True

    def plot() -> None:
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

    def where_to_rows(arr: np.ndarray) -> np.ndarray:
        """
        Takes in an image and returns a 2D array where each item represents a row in the image and each
        value in the array is the index in the image where the pixel is True
        :Example

        arr = [[1,0,0,1],
               [0,0,1,0],
               [0,0,0,1],
               [0,1,1,0]]
        height = 5

        returns [[0, 3], [2], [3], [1, 2]]

        :param arr: array to be manipulated
        :return: array of values that I can't be bothered to explain
        """
        a = np.array([*np.where(arr)]).transpose()
        b = np.where(np.append(0, np.diff(a[:, 0])) > 0)
        return np.split(a[:, 1], *b)

    def match_pixels(arr1, arr2, t_, match, threshold=0.01) -> np.ndarray:
        """
        :param arr1:
        :param arr2:
        :param t_:
        :param match:
        :param threshold:
        :return: [row or column that the point pair is, index of arr1, index of arr2]
        """
        arr1_all = where_to_rows(arr1)
        rows = np.array(*np.where(np.sum(pix_top_left, axis=1) > 0))
        print(rows)
        for i in range(rows.shape[0]):
            arr2_current = np.array(*np.where(arr2[rows[i]] > 0))
            for n in arr1_all[i]:
                if n > arr2_current[-1]:
                    break
                else:
                    for m in arr2_current:
                        if m > n:
                            xp = np.linspace(0, 1, m - n + 1)
                            fp = parent_image[rows[i], n:m + 1]
                            potential = mse(np.interp(t_, xp, fp), match)
                            if potential < threshold:
                                # Row/Column Found
                                return np.array([rows[i], n, m - n])

    path = "./training/micro.jpg"
    # path = "./training/test1.jpg"
    parent_image, parent_dims, sub_image, offset, size = setup_image(path, retoffset=True, retsize=True)

    print(f'Offset | {offset}')
    print(f'Size   | {size}')

    t0 = time()

    sub_dims = sub_image.shape

    # Possible corner pixel locations
    corner_colors = np.array([sub_image[0, 0], sub_image[0, -1], sub_image[-1, 0], sub_image[-1, -1]])
    pix_top_left = np.equal(parent_image, corner_colors[0])
    pix_top_right = np.equal(parent_image, corner_colors[1])
    pix_bottom_left = np.equal(parent_image, corner_colors[2])
    pix_bottom_right = np.equal(parent_image, corner_colors[3])

    if not check_params(): return False

    plot()

    for i in range(3):
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

    if not check_params(): return False

    plot()

    # Find the row or column with the least possible number of combinations
    s1 = np.sum(pix_top_left)
    s2 = np.sum(pix_top_right)
    s3 = np.sum(pix_bottom_left)
    s4 = np.sum(pix_bottom_right)

    # top row, bottom row, left column, right column
    offset = []
    size = [0, 0]
    min_line = np.array([s1 * s2, s3 * s4, s1 * s3, s2 * s4]).argmin()
    min_line = 1

    if min_line == 0:
        # top row
        t = np.linspace(0, 1, sub_dims[1])
        top_row_pair = match_pixels(pix_top_left, pix_top_right, t, sub_image[0])
        offset = [top_row_pair[0], top_row_pair[1]]
        size[1] = top_row_pair[2]
    elif min_line == 1:
        # bottom row
        t = np.linspace(0, 1, sub_dims[1])
        pair = match_pixels(pix_bottom_left, pix_bottom_right, t, sub_image[-1])
        print(pair)
        pass
    elif min_line == 2:
        # left column
        t = np.linspace(0, 1, sub_dims[0])
        pass
    else:
        # right column
        t = np.linspace(0, 1, sub_dims[0])
        pass

    t0 = time() - t0
    plot()
    return Result(offset, size, t0, [s1, s2, s3, s4])


if __name__ == "__main__":
    res = point_based()
    if isinstance(res, Result):
        print(res)
    else:
        print('Sub-image not located\nTry again with a larger threshold value.')
