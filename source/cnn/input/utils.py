import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
import scipy.stats as stats
import seaborn as sns
sns.set(style="white", color_codes=True)


def format_filename(s):
    """
    Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.

    Note: this method may produce invalid file names such as ``, `.` or `..`

    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    return filename


def envi2world(pointmatrix_local, GT):
    """
    convert the image coordinates (relative to pixel center) in pointmatrix_local to the
    (usually projected) world coordinates of envihdrfilename.
    pointmatrix_local: 2-column np.matrix [[x, y]]
    """

    # transform to hyperspectral figure coordinates
    P = pointmatrix_local[:, 0] + 0.5  # relative to pixel corner
    L = pointmatrix_local[:, 1] + 0.5
    xy = np.column_stack((GT[0] + P * GT[1] + L * GT[2],
                          GT[3] + P * GT[4] + L * GT[5]))

    return xy


def world2envi(pointmatrix, GT):
    """
    convert the (usually projected) world coordiates in pointmatrix to the image
    coordinates of envihdrfilename (relative to pixel center).
    pointmatrix: 2-column np.matrix [[x, y]]
    """

    # transform to hyperspectral figure coordinates
    X = pointmatrix[:, 0]
    Y = pointmatrix[:, 1]
    D = GT[1] * GT[5] - GT[2] * GT[4]
    xy = np.column_stack(((X * GT[5] - GT[0] * GT[5] + GT[2] * GT[3] - Y * GT[2]) / D - 0.5,
                          (Y * GT[1] - GT[1] * GT[3] + GT[0] * GT[4] - X * GT[4]) / D - 0.5))
    return xy


def envi2world_p(xy, GT):
    """
    convert one point from envi to world coordinate
    :param xy:
    :param GT:
    :return:
    """
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
    random_state = 797  # 646, 919, 390, 101
    # random_state = np.random.randint(100, 1000)
    print('Random seed:', random_state)
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
        nonzero_label = labels[labels > 0]
        print('{} :Mean = {}, Std = {}'.format(label_names[i], np.mean(nonzero_label), np.std(nonzero_label)))
        resize_img(name, 2000)

    print('-------Done visualizing labels--------')


def plot_pred_vs_target(x, y, color, name, save_path, epoch):
    """
    Scatter plot of prediction and target values together with histogram distribution
    of the two axes x and y
    :param x: target values
    :param y: corresponding predicted values
    :param color: color to plot
    :param name: task name
    :param save_path: saving path
    :param epoch:
    :return:
    """
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, s=2, c=color)
    # ax.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.7, label='Ideal prediction')
    # plt.ylim(top=1.3, bottom=-0.2)
    # ax.set_xlabel('Target')
    # ax.set_ylabel('Prediction')
    # ax.set_title('Task {}'.format(name))
    # ax.legend()
    # fig.savefig('{}/task_{}_e{}'.format(save_path, name, epoch))
    # plt.close(fig)

    g = sns.JointGrid(x, y, height=10, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    # g = sns.jointplot(x, y, kind='reg')
    g = g.plot_joint(plt.scatter, color=color, s=30, edgecolor="white")
    try:
        g = g.plot_marginals(sns.distplot, kde=True, color=color)
    except Exception as e:
        print("Encountered error when plotting join plot. Error: " + str(e))
        print("Predicted values: ", y)
        # g = g.plot_marginals(sns.distplot, kde=False, color=color)

    g = g.annotate(stats.pearsonr)
    g.set_axis_labels(xlabel='Target', ylabel='Prediction')

    g.ax_joint.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    g.savefig('{}/task_{}_e{}.png'.format(save_path, name, epoch))

    # hexbin plot
    g2 = sns.jointplot(x, y, height=10, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='hex', color=color, gridsize=50)
    g2.set_axis_labels(xlabel='Target', ylabel='Prediction')
    g2.ax_joint.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    g2.savefig('{}/task_{}_hexbin_e{}.png'.format(save_path, name, epoch))

    # kernel density estimation (kde) plot
    try:
        g3 = sns.jointplot(x, y, height=10, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='kde', color=color)
        g3.set_axis_labels(xlabel='Target', ylabel='Prediction')
        g3.ax_joint.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
        g3.savefig('{}/task_{}_kde_e{}.png'.format(save_path, name, epoch))
    except Exception as e:
        print("Encountered error when plotting kde. Error: " + str(e))
    plt.close('all')


def plot_largest_error_patches(task_label, topk_points, patch_size, task_name, path, epoch):
    """
    Plot top x% error patches on the map
    :param task_label: a randomly chosen task label (used for visualizing purpose, better used the rgb map)
    :param topk_points: list of points with highest errors
    :param patch_size: patch size
    :param task_name: name of the task
    :param path: saving path
    :param epoch:
    :return:
    """
    x = topk_points[:, 1] - patch_size // 2
    y = topk_points[:, 0] - patch_size // 2
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.grid(False)
    ax.set_title('Top 10% errors of {}'.format(task_name))
    im = ax.imshow(task_label, cmap='viridis', aspect='auto')
    plt.axis('off')
    # ax.scatter(x, y, s=50, c='r')
    for i in range(len(x)):
        rect = patches.Rectangle((x[i], y[i]), width=patch_size, height=patch_size,
                                 edgecolor='r', linewidth=2, facecolor='none')
        ax.add_patch(rect)
    # ax.plot(x_points, y_points, 'ro')
    save_name = '{}/topk_{}_e{}'.format(path, task_name, epoch)
    fig.savefig(save_name)
    plt.close(fig)
    # resize_img(save_name, 2500)


def plot_error_histogram(errors, bins, task_name, epoch, path):
    """
    Plot error histogram of the validation errors
    :param errors: list of errors, each corresponding to one prediction
    :param bins: number of bins to plot
    :param task_name: name of the current task being plotted
    :param epoch:
    :param path: saving path
    :return:
    """
    # test distribution:
    # distributions = ['norm', 'expon', 'gumbel', 'logistic']
    # for dist in distributions:
    #     result = stats.anderson(errors.numpy(), dist=dist)
    #     print('Result for testing %s distribution (%s): %s' % (dist, task_name, result))
    #     # for i in range(len(result.critical_values)):
    #     #     sl, cv = result.significance_level[i], result.critical_values[i]
    #     #     if result.statistic < result.critical_values[i]:
    #     #         print('Significant level %.3f: %.3f, data follows %s distribution' % (sl, cv, dist))
    #     #     else:
    #     #         print('Significant level %.3f: %.3f, data does not follow %s distribution' % (sl, cv, dist))
    #     # print('---End test---')
    # ######
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')

    counts, bins, _ = ax1.hist(errors, bins, density=False, facecolor='g', edgecolor='w', alpha=0.8)

    freq_cumsum = np.cumsum(counts)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative sum of frequency')
    ax2.plot(bins[1:], freq_cumsum, c='b')
    ax2.set_ylim(bottom=0, top=None)  # set this after plotting the data

    # err_cumsum = []
    # for b in bins:
    #     err_cumsum.append(torch.sum(errors[errors < b]))
    # ax2.plot(bins, err_cumsum, c='r')
    save_path = '{}/err_hist_{}_e{}'.format(path, task_name, epoch)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


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


def export_error_points(coords, rmsq_errors, geotrans, sum_errors, names, epoch, save_path):
    """
    Export points with corresponding errors to text file
    :param coords: point coordinates in image system (row and column)
    :param rmsq_errors: list root mean squared errors for each point
    :param geotrans: geo-transform mapping
    :param sum_errors: sum of errors for all tasks
    :param names: list of task names
    :param epoch:
    :param save_path:
    :return:
    """
    points_worldcoord = envi2world(coords, geotrans)
    file_data = []
    header = []
    ids = np.array(range(len(coords))).reshape(-1, 1)
    file_data.append(ids)
    file_data.append(points_worldcoord)
    file_data.append(rmsq_errors)
    file_data.append(sum_errors.unsqueeze(-1))
    file_data = np.concatenate(file_data, axis=1)
    header = np.append(header, ['#ID', 'X', 'Y'])
    header = np.append(header, names)
    header = np.append(header, 'sum_errors')
    header = '\t'.join(header)
    fname = '{}/error_world_coords_e{}.txt'.format(save_path, epoch)
    fmt = '\t'.join(['%i', '%i', '%i'] + ['%1.4f'] * (rmsq_errors.shape[-1] + 1))
    np.savetxt(fname, file_data, fmt=fmt, delimiter='\t', header=header, comments='')


def compute_data_distribution(labels, dataset, categorical):
    """
    Compute data distribution in training and validation sets
    :param labels: processed target labels
    :param dataset: training or validation set
    :param categorical: dictionary contains class info (equivalent to  metadata['categorical'])
    :return:
    """
    if labels.nelement() == 0:
        return []
    # TODO: make sure training and validation sets include all classes for each classification tasks
    data_labels = []  # labels of data points in the dataset
    for (r, c) in dataset:
        data_labels.append(labels[r, c, :])

    data_labels = torch.stack(data_labels, dim=0)
    weights = []
    num_classes = 0
    for idx, (key, values) in enumerate(categorical.items()):
        count = len(values)
        indices = torch.argmax(data_labels[:, num_classes:(num_classes + count)], dim=-1)
        unique_values, unique_count = np.unique(indices, return_counts=True)
        percentage = unique_count / np.sum(unique_count)
        # class_weight = torch.from_numpy(1 - percentage).float()
        # class_weight = torch.from_numpy(np.max(percentage) / percentage).float()

        # inverse median frequency
        median = np.median(percentage)
        class_weight = torch.from_numpy(median / percentage).float()
        # weights.append(class_weight)

        # different method to calculate class weights
        # class_weight = torch.from_numpy(np.log(0.7 * np.sum(unique_count) / unique_count)).float()
        # class_weight[class_weight < 1] = 1.0
        weights.append(class_weight)

        num_classes += count

        print('Class weight:', class_weight)
        print('Dataset distribution for task {}: classes={}, percentage={}, count={}'.format(
            key,
            values[unique_values],
            " ".join(map("{:.2f}%".format, 100 * percentage)),
            unique_count
        ))
    return weights
