import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt


def envi2world_p(xy, GT):
    P = xy[0] + 0.5  # relative to pixel corner
    L = xy[1] + 0.5
    xy = (GT[0] + P * GT[1] + L * GT[2],
          GT[3] + P * GT[4] + L * GT[5])
    return xy


def world2envi_p(xy, GT):
    X = xy[0]
    Y = xy[1]
    D = GT[1] * GT[5] - GT[2] * GT[4]
    xy = ((X * GT[5] - GT[0] * GT[5] + GT[2] * GT[3] - Y * GT[2]) / D - 0.5,
          (Y * GT[1] - GT[1] * GT[3] + GT[0] * GT[4] - X * GT[4]) / D - 0.5)
    # convert coordinates to integers
    xy = (int(np.round(xy[0])), int(np.round(xy[1])))
    return xy


def hyp2for(xy0, hgt, fgt):
    """
    Convert coordinates of a point in hyperspectral image system to forest map system
    :param xy0: coordinate of the point to convert, in order (col, row)
    :param hgt: hyperspectral geo transform
    :param fgt: forest geo transform
    :return: coordinate in forest map, in order (col, row)
    """
    xy1 = envi2world_p(xy0, hgt)
    xy2 = world2envi_p(xy1, fgt)
    return int(np.floor(xy2[0] + 0.1)), int(np.floor(xy2[1] + 0.1))


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
    :param tensor:
    :param row:
    :param col:
    :param patch_size:
    :return:
    """
    row1, col1 = row - patch_size // 2, col - patch_size // 2
    row2, col2 = row + patch_size // 2, col + patch_size // 2

    return tensor[row1:(row2 + 1), col1:(col2 + 1)]


def split_data(rows, cols, norm_inv, patch_size, stride=1, mode='grid'):
    """
    Split dataset into train, test, val sets based on the coordinates
    of each pixel in the hyperspectral image
    :param rows: number of rows in hyperspectral image
    :param cols: number of columns in hyperspectral image
    :param patch_size: patch_size for a training image patch, expected to be an odd number
    :param stride: amount of pixels to skip while looping through the hyperspectral image
    :param mode: sampling mode, if
                'random': randomly sample the pixels
                'split': reserve part of the input image for validation,
                'grid': sliding through the hyperspectral image without overlapping each other.
    :return: train, test, val lists with the pixel positions
    """
    train = []
    val = []
    coords = []
    if mode == 'grid':
        stride = patch_size  # overwrite striding value so that the patches are not overlapping
    # reserve 20% in the middle part of the hyperspectral image for validation
    val_row_start = round(rows * 3 / 5)
    val_row_end = val_row_start + round(rows / 6)
    for i in range(patch_size // 2, rows - patch_size // 2, stride):
        for j in range(patch_size // 2, cols - patch_size // 2, stride):
            patch = get_patch(norm_inv, i, j, patch_size)
            if torch.min(patch) > 0:  # make sure there is no black pixels in the patch
                if mode == 'random' or mode == 'grid':
                    coords.append((i, j))
                elif mode == 'split':
                    if i <= val_row_start - patch_size // 2 or val_row_end + patch_size // 2 <= i:
                        train.append((i, j))
                    elif val_row_start <= i <= val_row_end:
                            val.append((i, j))

    if mode == 'random' or mode == 'grid':
        # train, test = train_test_split(coords, train_size=0.8, random_state=123, shuffle=True)
        # train, val = train_test_split(train, train_size=0.9, random_state=123, shuffle=True)
        train, val = train_test_split(coords, train_size=0.9, random_state=123, shuffle=True)
    elif mode == 'split':
        np.random.seed(123)
        np.random.shuffle(train)
        np.random.seed(123)
        np.random.shuffle(val)

    print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    print('Train', train[0:10])
    print('Val', val[0:10])
    return train, val


def visualize_label(hyper_labels):
    _, _, B = hyper_labels.shape
    flatten_labels = hyper_labels.reshape(-1, B)
    for i in range(B):
        y_values = flatten_labels[:, i]
        plt.bar(len(flatten_labels), y_values)

    plt.savefig('./data/labels-bar.jpg')
    # TODO: pie charts for classes
    plt.show()


def open_as_rgb(hyper_image, wavelength, options):
    # if not visualizer:
    #     return

    i_r = abs(wavelength - 0.660).argmin()  # red band, the closest to 660 nm
    i_g = abs(wavelength - 0.550).argmin()  # green, closest band to 550 nm
    i_b = abs(wavelength - 0.490).argmin()  # blue, closest to 490 nm

    hyp_rgb = hyper_image[:, :, [i_r, i_g, i_b]] / 8
    hyp_rgb = np.asarray(np.copy(hyp_rgb).transpose((2, 0, 1)), dtype='float32')
    # visualizer.image(hyp_rgb)
    # Image.fromarray(hyp_rgb.astype('uint8')).save('./checkpoint/{}'.format(options.save_dir))
