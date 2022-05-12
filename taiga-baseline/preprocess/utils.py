import os
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import torch
import matplotlib
from scipy.ndimage import convolve

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
import scipy.stats as stats
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from tools.hypdatatools_img import get_geotrans

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


def split_data(rows, cols, mask, hyper_labels_cls, hyper_labels_reg, patch_size, stride=1, mode='grid', is_mask = False):
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
    test = []
    origin = []
    coords = []
    random_state = 797  # 646, 919, 390, 101
    R, C, _ = mask.size()
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
                    coords.append([i, j, 0, None, None])
                # elif mode == 'split':
                #     if i <= val_row_start - patch_size // 2 or val_row_end + patch_size // 2 <= i:
                #         train.append((i, j))
                #     elif val_row_start <= i <= val_row_end:
                #         val.append((i, j))

#    for i in range(patch_size, rows - patch_size, 1):
#        for j in range(patch_size, cols - patch_size, 1):
#            origin.append([i, j, 0, None, None])

    if mode == 'random' or mode == 'grid':
        if is_mask == True:
            train, test = train_test_split(coords, train_size=0.99, random_state=random_state, shuffle=True)
            train, val = train_test_split(train, train_size=0.99, random_state=random_state, shuffle=True)
        else:
            train, test = train_test_split(coords, train_size=0.8, random_state=random_state, shuffle=True)
            train, val = train_test_split(train, train_size=0.9, random_state=random_state, shuffle=True)
        # train, val = train_test_split(coords, train_size=0.8, random_state=random_state, shuffle=True)
    # elif mode == 'split':
    #     np.random.seed(random_state)
    #     np.random.shuffle(train)
    #     np.random.seed(random_state)
    #     np.random.shuffle(val)

    # augmentation = ['flip', 'radiation_noise', 'mixture_noise']
    augmentation = []
    new_data = []
    # augmentation code: 1: flip horizontally, 2: flip vertically, 3:radiation, 4: mixture
    if len(augmentation):
        for sample in train:
            r, c, _, _, _ = sample
            for aug in augmentation:
                if aug == 'flip':
                    # code = 1 if np.random.random() > 0.5 else 2
                    code = 1
                    new_data.append([r, c, code, None, None])
                elif aug == 'radiation_noise' and np.random.random() < 0.15:
                    code = 3
                    new_data.append([r, c, code, None, None])
                elif aug == 'mixture_noise' and np.random.random() < 0.3:
                    code = 4
                    for trial in range(15):
                        r_row, r_col = np.random.randint(-patch_size * 5, patch_size * 5, size=2)
                        row2, col2 = r + r_row, c + r_col
                        if not in_hypmap(C, R, col2, row2, patch_size):
                            trial += 1
                            continue
                        patch = get_patch(mask, row2, col2, patch_size)
                        if torch.min(patch) > 0:
                            tgt_cls1 = hyper_labels_cls[r, c]
                            tgt_cls2 = hyper_labels_cls[row2, col2]
                            tgt_reg1 = hyper_labels_reg[r, c]
                            tgt_reg2 = hyper_labels_reg[row2, col2]
                            # if torch.all(torch.eq(tgt_cls1, tgt_cls2)) == 1 and torch.all(torch.eq(tgt_reg1, tgt_reg2)) == 1:
                            if np.array_equal(tgt_cls1, tgt_cls2) and np.array_equal(tgt_reg1, tgt_reg2):
                                new_data.append([r, c, code, row2, col2])
                                break

    train = train + new_data  # concat old and new data
    np.random.seed(123)
    np.random.shuffle(train)
    train = np.array(train)
    val = np.array(val)
    print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    print('Train', train[0:10])
    print('Val', val[0:10])
    return train, test, val, coords


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
        #print('{} : Mean = {}, Std = {}'.format(label_names[i], np.mean(nonzero_label), np.std(nonzero_label)))
        resize_img(name, 2000)

    #print('-------Done visualizing labels--------')
    print('Visualization saved to %s' % save_path)


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
    #print(np.max(hyp_rgb))
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
    #img.thumbnail((width, height), Image.ANTIALIAS)
    img.save('%s/rgb_image.png' % path)
    print("Hyperspectral image saved in RGB format as %s/rgb_image.png" % path)


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
    print('start getting data labels')
    data_labels = []  # labels of data points in the dataset
    for (r, c, _, _, _) in dataset:
        data_labels.append(labels[r, c, :])
    print('end getting data labels')

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


def get_device(id):
    device = torch.device('cpu')
    if id > -1 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(id))
    print("Number of GPUs available %i" % torch.cuda.device_count())
    print("Training on device: %s" % device)
    return device


def compute_accuracy(predict, tgt, categorical):
    """
    Return number of correct prediction of each tgt label
    :param predict: tensor of predicted outputs
    :param tgt: tensor of ground truth labels
    :return: number of correct predictions for every single classification task
    """

    # reshape tensor in (*, n_cls) format
    # this is mainly for LeeModel that output the prediction for all pixels
    # from the source image with shape (batch, patch, patch, n_cls)
    predict = predict.cpu()
    tgt = tgt.cpu()

    n_cls = tgt.shape[-1]
    predict = predict.view(-1, n_cls)
    tgt = tgt.view(-1, n_cls)
    #####
    pred_indices = []
    tgt_indices = []
    n_correct = torch.tensor([0.0] * len(categorical))
    num_classes = 0
    for idx, (key, values) in enumerate(categorical.items()):
        count = len(values)
        pred_class = predict[:, num_classes:(num_classes + count)]
        tgt_class = tgt[:, num_classes:(num_classes + count)]
        pred_index = pred_class.argmax(-1)  # get indices of max values in each row
        tgt_index = tgt_class.argmax(-1)
        pred_indices.append(pred_index)
        tgt_indices.append(tgt_index)
        true_positive = torch.sum(pred_index == tgt_index).item()
        n_correct[idx] += true_positive
        num_classes += count

    pred_indices = torch.stack(pred_indices, dim=1)
    tgt_indices = torch.stack(tgt_indices, dim=1)
    return n_correct, pred_indices, tgt_indices


def remove_ignored_tasks(hyper_labels, options, metadata):
    hyper_labels_cls = torch.tensor([], dtype=hyper_labels.dtype)
    hyper_labels_reg = torch.tensor([], dtype=hyper_labels.dtype)
    start = 0
    categorical = metadata['categorical'].copy()

    valid_indices = np.array(options.ignored_cls_tasks)
    valid_indices = valid_indices[valid_indices < len(metadata['cls_label_names'])]
    metadata['cls_label_names'] = np.delete(metadata['cls_label_names'], valid_indices)
    for idx, (key, values) in enumerate(categorical.items()):
        n_classes = len(values)
        if idx not in options.ignored_cls_tasks:
            hyper_labels_cls = torch.cat((hyper_labels_cls, hyper_labels[:, :, start:(start + n_classes)]), 2)
        else:
            del metadata['categorical'][key]
            metadata['num_classes'] -= n_classes
        start += n_classes

    valid_indices = np.array(options.ignored_reg_tasks)
    valid_indices = valid_indices[valid_indices < len(metadata['reg_label_names'])]
    metadata['reg_label_names'] = np.delete(metadata['reg_label_names'], valid_indices)
    for idx in range(start, hyper_labels.shape[-1]):
        true_idx = idx - start
        if true_idx not in options.ignored_reg_tasks:
            hyper_labels_reg = torch.cat((hyper_labels_reg, hyper_labels[:, :, idx:(idx + 1)]), 2)
        else:
            del metadata['regression'][true_idx]  # normal python dict

    return hyper_labels_cls, hyper_labels_reg


def multiclass_roc_auc_score(y_true, y_pred, average="macro"):
    lb = LabelBinarizer()

    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average=average)


def compute_cls_metrics(pred_cls_indices, tgt_cls_indices, options, categorical, mode='validation'):
    conf_matrices = []
    balanced_accuracies = []
    task_accuracies = []
    avg_accuracy = 0.0
    if not options.no_classification:
        task_accuracies, pred_cls_indices, tgt_cls_indices = compute_accuracy(pred_cls_indices, tgt_cls_indices, categorical)
        task_accuracies = task_accuracies * 100 / len(tgt_cls_indices)
        avg_accuracy = torch.mean(task_accuracies)

        print('--%s metrics--' % mode.capitalize())
        for i in range(tgt_cls_indices.shape[-1]):
            conf_matrix = confusion_matrix(tgt_cls_indices[:, i], pred_cls_indices[:, i])
            # convert to percentage along rows
            conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix = np.around(100 * conf_matrix, decimals=2)
            conf_matrices.append(conf_matrix)
            label_accuracy = np.around(np.mean(conf_matrix.diagonal()), decimals=2)
            balanced_accuracies.append(label_accuracy)
            print('---Task %s---' % i)
            precision = precision_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
            recall = recall_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
            f1 = f1_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
            auc_score = multiclass_roc_auc_score(tgt_cls_indices[:, i], pred_cls_indices[:, i])
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1 score', f1)
            print('ROC-AUC score:', auc_score)

        avg_balanced_accuracy = np.around(np.mean(balanced_accuracies), decimals=2)
        print('Average balanced accuracy=%s, task balanced accuracies=%s'
              % (avg_balanced_accuracy, balanced_accuracies))
        print('-------------')
    return balanced_accuracies, avg_accuracy, task_accuracies, conf_matrices


def compute_reg_metrics(dataset_loader, all_pred_reg, all_tgt_reg, epoch, options, metadata, hyper_labels_reg, save_path,
                        should_save, mode='validation'):
    """
    Compute related regression metrics
    :param dataset_loader: data loader of the dataset (should be 'val_loader' or 'test_loader')
    :param all_pred_reg: raw prediction of the whole data loader
    :param all_tgt_reg: target values
    :param epoch: epoch of the current model
    :param options: 
    :param metadata: 
    :param hyper_labels_reg: 
    :param save_path: 
    :param should_save: 
    :param mode: 
    :return: 
    """
    N_samples = len(dataset_loader)
    hypGt = get_geotrans(options.hyper_data_header)
    if not options.no_regression:
        absolute_errors = torch.abs(all_pred_reg - all_tgt_reg)
        signed_errors = all_pred_reg - all_tgt_reg
        mae_error_per_task = torch.mean(absolute_errors, dim=0)
        print('{} MAE={:.5f}, MAE per task={}'.format(
            mode.capitalize(),
            torch.mean(mae_error_per_task),
            " ".join(map("{:.5f}".format, mae_error_per_task.data.cpu().numpy()))
        ))
        absolute_errors_all = torch.sum(absolute_errors, dim=1)
        signed_errors_all = torch.sum(signed_errors, dim=1)

        if should_save:
            n_reg = all_pred_reg.shape[-1]
            coords = np.array(dataset_loader.dataset.coords)

            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, n_reg)]
            names = metadata['reg_label_names']

            plot_error_histogram(absolute_errors_all, 100, 'all_tasks', epoch, save_path)
            plot_error_histogram(signed_errors_all, 100, 'all_tasks_signed_errors', epoch, save_path)

            k = N_samples // 10  # 10% of the largest error
            value, indices = torch.topk(absolute_errors_all, k, dim=0, largest=True, sorted=False)

            topk_points = coords[indices]
            task_label = hyper_labels_reg[:, :, 0]
            # chose the first task label just for visualization
            plot_largest_error_patches(task_label, topk_points, dataset_loader.dataset.patch_size,
                                       'all_tasks', save_path, epoch)

            export_error_points(coords, absolute_errors, hypGt, absolute_errors_all, names, epoch, save_path)

            # pred = np.zeros((dataset_loader.dataset.hyper_row, dataset_loader.dataset.hyper_col))
            # print('start get pred')
            # for i in range(dataset_loader.dataset.coords.shape[0]):
            #     if i < all_pred_reg.shape[0]:
            #         pred[dataset_loader.dataset.coords[i][0], dataset_loader.dataset.coords[i][1]] = all_pred_reg[i][0]
            # print('done get pred')
            # compute_variance_pred_neighborhoods(pred, 'all_tasks', save_path, epoch, 3)

            for i in range(n_reg):
                x, y = all_tgt_reg[:, i], all_pred_reg[:, i]

                plot_pred_vs_target(x, y, colors[i], names[i], save_path, epoch)

                # plot error histogram
                absolute_errors = torch.abs(x - y)
                plot_error_histogram(absolute_errors, 100, names[i], epoch, save_path)
                plot_error_histogram(signed_errors, 100, names[i] + '_signed_errors', epoch, save_path)

                # plot top k largest errors on the map
                task_label = hyper_labels_reg[:, :, i]
                value, indices = torch.topk(absolute_errors, k, dim=0, largest=True, sorted=False)

                topk_points = coords[indices]
                plot_largest_error_patches(task_label, topk_points, dataset_loader.dataset.patch_size,
                                           names[i], save_path, epoch)

                # compute_variance_pred_neighborhoods(y, names[i], save_path, epoch, 5)

def compute_variance_pred_neighborhoods(task_label, task_name, path, epoch, kernel_size=3):
    out = np.ones(task_label.shape)

    height, width = task_label.shape[:2]
    print('start get neighbors')
    for i in range(height):
        for j in range(width):
            out[i, j] = get_neighbors(task_label, i, j, kernel_size).flatten().var()
    print('done get neighbors')

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.grid(False)
    ax.set_title('Variance of the {} output with {}x{} neighborhoods'.format(task_name, kernel_size, kernel_size))
    im = ax.imshow(out, cmap='viridis', aspect='auto')
    plt.axis('off')
    save_name = '{}/variance_{}x{}_{}_e{}'.format(path, kernel_size, kernel_size, task_name, epoch)
    fig.savefig(save_name)
    plt.close(fig)

def get_neighbors(arr, x, y, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be an odd value')

    half = kernel_size // 2
    return arr[max(0, x - half):min(x + kernel_size - half, arr.shape[0]), max(0, y - half):min(y + kernel_size - half, arr.shape[1])]
