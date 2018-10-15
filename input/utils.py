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


def split_data(labels, rows, cols, patch_size):
    """
    Split dataset into train, test, val sets based on the coordinates
    of each pixel in the hyperspectral image
    :param labels: labels for hyperspectral image
    :param rows: number of rows in hyperspectral image
    :param cols: number of columns in hyperspectral image
    :param patch_size: patch_size for a training image patch, expected to be an odd number
    :return: train, test, val lists with the pixel positions
    """
    # mask = np.sum(labels.numpy(), axis=2)
    # assign 0 to all pixels around the edges
    # pad = patch_size // 2
    # mask[:pad, :] = 0
    # mask[(rows - pad+1):rows, :] = 0
    # mask[:, :pad] = 0
    # mask[:, (cols - pad + 1):cols] = 0
    # x_pos, y_pos = np.nonzero(mask)
    # coords = [(x, y) for x, y in zip(x_pos, y_pos)]
    coords = []
    for i in range(patch_size // 2, rows - patch_size // 2):
        for j in range(patch_size // 2, cols - patch_size // 2):
            coords.append((i, j))
            # if np.sum(labels.numpy()[:, :, 0:31] != 0):  # very expensive operation
            #     coords.append((i, j))

    train, test = train_test_split(coords, train_size=0.8, random_state=123, shuffle=True)
    train, val = train_test_split(train, train_size=0.9, random_state=123, shuffle=True)
    # np.savetxt('train.txt', train, fmt="%d")
    # np.savetxt('test.txt', test, fmt="%d")
    # np.savetxt('val.txt', val, fmt="%d")

    print('Number of pixels: ', len(coords))
    return train, test, val
