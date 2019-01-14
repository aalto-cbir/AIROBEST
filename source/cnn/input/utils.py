import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import string


def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.

Note: this method may produce invalid file names such as ``, `.` or `..`

"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    return filename


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


def split_data(rows, cols, mask, patch_size, stride=1, mode='grid'):
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
    random_state = 101
    if mode == 'grid':
        stride = patch_size  # overwrite striding value so that the patches are not overlapping
    # reserve 20% in the middle part of the hyperspectral image for validation
    val_row_start = round(rows * 3 / 5)
    val_row_end = val_row_start + round(rows / 6)
    for i in range(patch_size // 2, rows - patch_size // 2, stride):
        for j in range(patch_size // 2, cols - patch_size // 2, stride):
            patch = get_patch(mask, i, j, patch_size)
            if torch.min(patch) > 0:  # make sure there is no black pixels in the patch
                if mode == 'random' or mode == 'grid':
                    coords.append((i, j))
                elif mode == 'split':
                    if i <= val_row_start - patch_size // 2 or val_row_end + patch_size // 2 <= i:
                        train.append((i, j))
                    elif val_row_start <= i <= val_row_end:
                        val.append((i, j))

    if mode == 'random' or mode == 'grid':
        # train, test = train_test_split(coords, train_size=0.8, random_state=random_state, shuffle=True)
        # train, val = train_test_split(train, train_size=0.9, random_state=random_state, shuffle=True)
        train, val = train_test_split(coords, train_size=0.8, random_state=random_state, shuffle=True)
    elif mode == 'split':
        np.random.seed(random_state)
        np.random.shuffle(train)
        np.random.seed(random_state)
        np.random.shuffle(val)

    print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    print('Train', train[0:10])
    print('Val', val[0:10])
    return train, val


def resize_img(path, threshold):
    assert os.path.exists(path), 'Image does not exists in path: %s' % path
    img = Image.open(path)
    width, height = img.size
    max_size = np.max([width, height])
    if max_size > threshold:
        scaling_factor = max_size // threshold + 1
        new_width, new_height = width // scaling_factor, height // scaling_factor
        img.thumbnail((new_width, new_height), Image.ANTIALIAS)
        img.save(path)


def visualize_label(target_labels, label_names, save_path):
    """
    Visualize normalized hyper labels
    :param target_labels:
    :param label_names:
    :param save_path:
    :return:
    """
    for i in range(target_labels.shape[-1]):
        labels = target_labels[:, :, i]
        name = '{}/{}.png'.format(save_path, label_names[i])

        # image with colorbar
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.grid(False)
        ax.set_title('Data distribution of %s' % label_names[i])
        im = ax.imshow(labels, cmap='viridis')
        plt.axis('off')
        plt.colorbar(im)
        fig.savefig(name)
        plt.close(fig)
        #####

        # plt.imsave(name, labels)
        print('{}: Min={}, Max={}, Mean={}'.format(label_names[i], np.min(labels), np.max(labels), np.mean(labels)))
        resize_img(name, 2000)

    print('-------Done visualizing labels--------')


def save_as_rgb(hyper_image, wavelength, path):
    """
    Save hyperspectral in rgb format
    :param hyper_image: hyperspectral image
    :param wavelength: list of wavelengths for each spectral band
    :param path: path to save the image
    :return:
    """
    i_r = abs(wavelength - 0.660).argmin()  # red band, the closest to 660 nm
    i_g = abs(wavelength - 0.550).argmin()  # green, closest band to 550 nm
    i_b = abs(wavelength - 0.490).argmin()  # blue, closest to 490 nm

    hyp_rgb = hyper_image[:, :, [i_r, i_g, i_b]] / 8  # heuristically scale down the reflectance for good representation
    print(np.max(hyp_rgb))
    # hyp_rgb[hyp_rgb > 255] = hyp_rgb[hyp_rgb > 255] / 5  # continue scaling down high values
    hyp_rgb[hyp_rgb > 255] = 255
    hyp_rgb[hyp_rgb == 0.0] = 255

    # hyp_rgb = hyp_rgb / 40
    # hyp_rgb = np.asarray(np.copy(hyp_rgb).transpose((2, 0, 1)), dtype='float32')
    height, width = hyp_rgb.shape[:2]
    threshold = 2000
    max_size = np.max([width, height])
    if max_size > threshold:
        scaling_factor = max_size // threshold + 1
        height = height // scaling_factor
        width = width // scaling_factor
    print("RGB size (wxh):", width, height)
    img = Image.fromarray(hyp_rgb.astype('uint8'))
    img.thumbnail((width, height), Image.ANTIALIAS)
    img.save('%s/rgb_image.png' % path)


def compute_data_distribution(labels, dataset, categorical):
    """

    :param labels:
    :param dataset:
    :param categorical: dictionary contains class info (equivalent to  metadata['categorical'])
    :return:
    """
    data_labels = []  # labels of data points in the dataset
    for (r, c) in dataset:
        data_labels.append(labels[r, c, :])

    data_labels = torch.stack(data_labels, dim=0)
    num_classes = 0
    for idx, (key, values) in enumerate(categorical.items()):
        count = len(values)
        indices = torch.argmax(data_labels[:, num_classes:(num_classes+count)])
        unique_values, unique_count = np.unique(indices, return_counts=True)
        percentage = unique_count / np.sum(unique_count)

        print('Dataset distribution for task {}: classes={}, percentage={}'.format(
            key,
            values[unique_values],
            " ".join(map("{:.2f}%".format, percentage))))
