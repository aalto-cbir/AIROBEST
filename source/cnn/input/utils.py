import numpy as np
from sklearn.model_selection import train_test_split


def in_hypmap(W, H, x, y, patch_size):
    """
    Check if a point with coordinate (x, y) lines within the image with size (W, H)
    in Cartesian coordinate system
    :param W: width of the image (columns)
    :param H: height of the image (rows)
    :param x:
    :param y:
    :return: True if it's inside
    """
    # TODO: patch_size is expected to be odd number
    x1, y1 = x - patch_size // 2, y - patch_size // 2
    x2, y2 = x + patch_size // 2, y + patch_size // 2
    return x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H


def get_patch(tensor, row, col, patch_size):
    """
    Get patch from center pixel
    :param row:
    :param col:
    :return:
    """
    row1, col1 = row - patch_size // 2, col - patch_size // 2
    row2, col2 = row + patch_size // 2, col + patch_size // 2

    return tensor[row1:(row2 + 1), col1:(col2 + 1)]


def split_data(rows, cols, norm_matrix, patch_size, step=1):
    """
    Split dataset into train, test, val sets based on the coordinates
    of each pixel in the hyperspectral image
    :param rows: number of rows in hyperspectral image
    :param cols: number of columns in hyperspectral image
    :param patch_size: patch_size for a training image patch, expected to be an odd number
    :param step: amount of pixels to skip while looping through the hyperspectral image
    :return: train, test, val lists with the pixel positions
    """
    train = []
    val = []
    # reserve 20% in the middle part of the hyperspectral image for validation
    val_row_start = round(rows * 2 / 5)
    val_row_end = val_row_start + round(rows / 5)
    for i in range(patch_size // 2, rows - patch_size // 2, step):
        for j in range(patch_size // 2, cols - patch_size // 2, step):
            patch = get_patch(norm_matrix, i, j, patch_size)
            if np.min(patch) > 0:  # make sure there is no white pixels in the patch
                if i <= val_row_start - patch_size // 2 or val_row_end + patch_size // 2 <= i:
                    train.append((i, j))
                elif val_row_start + patch_size // 2 <= i <= val_row_end - patch_size // 2:
                    val.append((i, j))

    # train, test = train_test_split(coords, train_size=0.8, random_state=123, shuffle=True)
    # train, val = train_test_split(train, train_size=0.9, random_state=123, shuffle=True)

    print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    return train, val
