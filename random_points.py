from setup import *
from functions import *
from time import time

if __name__ == "__main__":
    parent_image, parent_dims, sub_image, offset, size = setup_image("./training/test1.jpg", retoffset=True, retsize=True)
    print(offset)
    t0 = time()
    # Top Row
    top_row_colors = np.array([sub_image[0, 0], sub_image[0, -1]])
    pix_top_left = np.equal(parent_image, top_row_colors[0])
    pix_top_right = np.equal(parent_image, top_row_colors[1])
    pix_top_row = np.greater_equal(np.add(pix_top_right, pix_top_left).sum(axis=1), 2)
    pix_top_row = np.multiply(pix_top_left, pix_top_row.reshape(pix_top_row.shape[0], 1))

    # Left Column
    bottom_left_color = sub_image[-1, 0]
    pix_bottom_left = np.equal(parent_image, bottom_left_color)
    pix_left_col = np.greater_equal(np.add(pix_top_left, pix_bottom_left).sum(axis=0), 2)
    pix_left_col = np.multiply(pix_top_left, pix_left_col)

    possible_top_left = np.multiply(pix_top_row, pix_left_col)

    # print(offset in np.array(np.where(pix_top_row == 1)).transpose())
    print(f'{round(1000 * (time() - t0), 3)}ms elapsed')
    print(f'{round(100 * (1 - np.mean(possible_top_left)), 3)}% excluded')
    print(offset in np.array([np.where(possible_top_left == 1)]).transpose())

    fig, ax = plt.subplots()
    ax.imshow(possible_top_left)
    ax.add_patch(rect((offset[1], offset[0]), size[1], size[0], edgecolor='red', facecolor='none'))
    plt.show()
