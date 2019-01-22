#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process hyperspectral data
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import spectral
import torch
import matplotlib.pyplot as plt

from input.utils import hyp2for, world2envi_p, save_as_rgb, visualize_label, split_data, format_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from tools.hypdatatools_img import get_geotrans

sys.stdout.flush()


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hyper_data_path',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/20170615_reflectance_mosaic_128b.hdr',
                        help='Path to hyperspectral data')
    parser.add_argument('-forest_data_path',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/forestdata.hdr',
                        help='Path to forest data')
    parser.add_argument('-human_data_path',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/Titta2013.txt',
                        help='Path to human verified data (Titta points)')
    parser.add_argument('-save_dir',
                        required=False, type=str,
                        default='mosaic',
                        help='Directory to save all generated files')
    parser.add_argument('-src_file_name',
                        required=False, type=str,
                        default='hyperspectral_src',
                        help='Save hyperspectral image with this name')
    parser.add_argument('-tgt_file_name',
                        required=False, type=str,
                        default='hyperspectral_tgt',
                        help='Save hyperspectral labels with this name')
    parser.add_argument('-metadata_file_name',
                        required=False, type=str,
                        default='metadata',
                        help='Save metadata of the processed data with this name')
    parser.add_argument('-normalize_method',
                        type=str,
                        default='l2norm_along_channel', choices=['l2norm_along_channel', 'l2norm_channel_wise'],
                        help='Normalization method for input image')
    parser.add_argument('-label_normalize_method',
                        type=str,
                        default='clip', choices=['minmax_scaling', 'clip'],  # TODO: support z-score
                        help='Normalization method for target labels')
    parser.add_argument('-categorical_bands', nargs='+',
                        required=False, default=[0, 1, 2, 9],
                        help='List of band indices in the target labels for categorical tasks')
    parser.add_argument('-ignored_bands', nargs='+',
                        required=False, default=[3, 8],
                        help='List of band indices in the target labels to ignore')
    parser.add_argument('-ignore_zero_labels', type=bool,
                        default=True,
                        help='Whether to ignore target labels with 0 values in data statistics')
    opt = parser.parse_args()
    opt.categorical_bands = list(map(int, opt.categorical_bands))
    opt.ignored_bands = list(map(int, opt.ignored_bands))
    return opt


def get_hyper_labels(hyper_image, forest_labels, hyper_gt, forest_gt):
    """
    Create hyperspectral labels from forest data
    :param hyper_image: W1xH1xC
    :param forest_labels: W2xH2xB
    :return:
    """
    rows, cols, _ = hyper_image.shape

    c, r = hyp2for((0, 0), hyper_gt, forest_gt)
    hyper_labels = np.array(forest_labels[r:(r+rows), c:(c+cols)])

    return hyper_labels


def in_hypmap(W, H, x, y):
    """
    Check if a point with coordinate (x, y) lines within the image with size (W, H)
    in Cartesian coordinate system
    :param W: width of the image (columns)
    :param H: height of the image (rows)
    :param x:
    :param y:
    :return: True if it's inside
    """
    return W >= x >= 0 and H >= y >= 0


def apply_human_data(human_data_path, hyper_labels, hyper_gt, forest_columns):
    """
    Improve hyperspectral data labels by applying data labels collected at certain points
    by human
    :param human_data_path:
    :param hyper_labels:
    :param hyper_gt:
    :param forest_columns:
    :return:
    """

    # dictionary contains mapping from column of Titta data (key)
    # to equivalent column (or 'band' to be more precise) in Forest data (value)
    # ex: items with key-value pair: (3, 9) means column 3 (index starts from
    # 'Fertilityclass' column) of Titta has the same data as column 9 of Forest data
    column_mappings = {
        0: 0,
        3: 9,
        4: 10,
        5: 6,
        6: 4,
        8: 7,
        11: 8,
        12: 16
    }

    df = pd.read_csv(human_data_path,
                     encoding='utf-16',
                     na_values='#DIV/0!',
                     skiprows=[187, 226],  # skip 2 rows with #DIV/0! values
                     delim_whitespace=True)

    data = df.as_matrix()
    coords = data[:, 1:3]
    labels = data[:, 3:]

    # scale data in columns 6 and 12 of Titta data to match
    # the value in Forest data. The value of these columns are
    # scaled up 100 times
    scaling_idx = [6, 12]
    labels[:, scaling_idx] = labels[:, scaling_idx] * 100
    ######

    titta_columns = df.columns.values[3:]
    print('Column mappings from Titta to Forest data')
    for titta_idx, forest_idx in column_mappings.items():
        print('%s --> %s ' % (titta_columns[titta_idx], forest_columns[forest_idx]))

    counter = 0
    for idx, (x, y) in enumerate(coords):
        c, r = world2envi_p((x, y), hyper_gt)

        if not in_hypmap(hyper_labels.shape[1], hyper_labels.shape[0], c, r):
            continue
        counter += 1
        label = labels[idx]
        for titta_idx, forest_idx in column_mappings.items():
            hyper_labels[r, c, forest_idx] = label[titta_idx]

    print('There are %d Titta points line inside the hyperspectral map.' % counter)
    return hyper_labels


def plot_chart(ax, label_name, unique_values, percentage):

    x = np.arange(len(unique_values))
    ax.bar(x, percentage)

    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar label using above list
    total = sum(totals)
    # set individual bar label using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x() + .12, i.get_height(), str(round((i.get_height() / total) * 100, 1)), fontsize=10,
                color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_values)
    plt.tight_layout()
    ax.set_title(label_name)


def handle_class_balancing(percentage, unique_values, threshold=5):
    minor_percentages = percentage[percentage < threshold]
    remaining_percentages = percentage[percentage >= threshold]
    minor_classes = unique_values[percentage < threshold]
    remaining_classes = unique_values[percentage >= threshold]
    # 2 approaches to handle class imbalance
    #   1: combine minor classes into a NEW common class (if minor_sum >= threshold)
    #   2: combine minor classes into smallest class above the threshold (otherwise)
    minor_sum = minor_percentages.sum()

    if minor_sum >= threshold:
        combined_class_idx = len(remaining_percentages)
        new_class = np.max(remaining_classes) + 1
        remaining_classes = np.append(remaining_classes, new_class)
        remaining_percentages = np.append(remaining_percentages, minor_sum)

    else:
        combined_class_idx = np.argmin(remaining_percentages)
        remaining_percentages[combined_class_idx] += minor_sum

    index_dict = dict([(val, idx) for (idx, val) in enumerate(remaining_classes)])

    # point removed classes to the combined_class_idx
    for i in minor_classes:
        index_dict[i] = combined_class_idx

    print('Removed classes: {}, percentages: {}'.format(minor_classes, minor_percentages))
    print('Final classes: {}, percentages: {}'.format(remaining_classes, remaining_percentages))
    print('Index dict:', index_dict)
    return index_dict, remaining_classes, remaining_percentages


def process_labels(labels, categorical_bands, ignored_bands, cls_label_names, reg_label_names, zero_count, save_path, options):
    """
    - Calculate class member for each categorical class
    - Sort class members and assign indices
    - Transform data to vector of 0 and 1 indicating which class and label
    the data belongs to
    Example:
        0 fertilityclass    1  2  3          ->  0  1  2
        2 maintreespecies   2  4  6          ->  3  4  5

    The label [2, 6] (fertiliticlass: 2, maintreespieces: 6) will become
    [0, 1, 0, 0, 0, 1]
    :param labels: shape RxCxB
    :param categorical_bands:
    :param ignored_bands:
    :param cls_label_names:
    :param zero_count:
    :param save_path:
    :return:
    """

    transformed_data = None
    num_classes = 0
    metadata = {}
    categorical = {}
    R, C, _ = labels.shape

    fig, axs = plt.subplots((len(categorical_bands) + 1) // 2, 2)

    for i, b in enumerate(categorical_bands):
        band = labels[:, :, b]
        unique_values, unique_counts = np.unique(band, return_counts=True)
        # recount zero values as there are zero pixel values from the hyperspectral image
        if unique_values[0] == 0.0:
            if options.ignore_zero_labels:
                unique_values = np.delete(unique_values, 0)
                unique_counts = np.delete(unique_counts, 0)
            else:
                unique_counts[0] -= zero_count
        percentage = 100 * unique_counts / np.sum(unique_counts)
        print('Band {} ({}): unique values = {}, original distribution = {}, frequency = {}'
              .format(b, cls_label_names[i], unique_values, " ".join(map("{:.2f}%".format, percentage)), unique_counts))

        # plot_chart(axs[i // 2, i % 2], label_names[i], unique_values, percentage)  # original distribution
        # index_dict = dict([(val, idx) for (idx, val) in enumerate(unique_values)])

        threshold = 5
        index_dict, new_unique_values, new_percentages = handle_class_balancing(percentage, unique_values, threshold)
        plot_chart(axs[i // 2, i % 2], cls_label_names[i], new_unique_values, new_percentages)
        categorical[b] = new_unique_values
        one_hot = np.zeros((R, C, len(new_unique_values)), dtype=int)
        for row in range(R):
            for col in range(C):
                if band[row, col] == 0 and options.ignore_zero_labels:
                    continue
                idx = index_dict[band[row, col]]  # get the index of the value  band[row, col] in one-hot vector
                one_hot[row, col, idx] = 1

        if transformed_data is None:
            transformed_data = one_hot
        else:
            transformed_data = np.concatenate((transformed_data, one_hot), axis=2)
        num_classes += len(new_unique_values)

    # delete all the categorical classes from label data
    labels = np.delete(labels, np.append(categorical_bands, ignored_bands), axis=2)

    fig.savefig('%s/class_distribution.png' % save_path)
    plt.close(fig)

    # normalize data for regression task
    normalized_labels = []
    reg_stats = {}
    for b in range(labels.shape[2]):
        band = labels[:, :, b]
        true_max = np.max(band)
        if options.label_normalize_method == 'minmax_scaling':
            max_val = true_max
            min_val = np.min(band)
        elif options.label_normalize_method == 'clip':
            value_matrix = band[band > 0] if options.ignore_zero_labels else band
            # use 2 and 98 percentile as thresholding values
            # TODO: clip values can be passes as arguments
            # NOTE: if we want to clip the matrix to a lower bound,
            # more sophisticated handling needs to take place as the original
            # zero data labels (road, lakes) will become negative => handle properly in
            # splitting train, val data sets
            max_val = np.percentile(value_matrix, 98)
            # min_val = np.percentile(value_matrix, 2)
            min_val = np.min(band)

            # clip the data with thresholding values
            band[band > max_val] = max_val

            # zero_mask = band == 0
            # band[band < min_val] = min_val  # this changes 0 values labels
            # band[zero_mask] = 0  # put back 0 values has been changed by min_val

        print('Regression {}: lower bound = {}, upper bound = {}, true_max = {}'
              .format(reg_label_names[b], min_val, max_val, true_max))
        reg_stats[b] = {'min': min_val, 'max': max_val, 'true_max': true_max}
        if max_val != min_val:
            band = (band - min_val) / (max_val - min_val)
        elif max_val != 0:  # if all items have the same non-zero value
            band.fill(0.5)
        else:  # if all are 0, if this happens, consider remove the whole band from data
            band.fill(0.0)
            print('Band with index %d has all zero values, consider removing it!' % b)

        normalized_labels.append(band)
    normalized_labels = np.stack(normalized_labels, axis=2)
    # concatenate with newly transformed data
    normalized_labels = np.concatenate((transformed_data, normalized_labels), axis=2)

    metadata['categorical'] = categorical
    metadata['num_classes'] = num_classes
    metadata['regression'] = reg_stats

    return normalized_labels, metadata


def main():
    print('Start processing data...')
    #######
    options = parse_args()
    print(options)

    save_path = './data/%s' % options.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hyper_data = spectral.open_image(options.hyper_data_path)
    hyper_gt = get_geotrans(options.hyper_data_path)
    hyper_image = hyper_data.open_memmap()
    hyper_image = np.array(hyper_image[:, :, 0:110])  # only take the first 110 spectral bands, the rest are noisy
    R, C, B = hyper_image.shape

    forest_data = spectral.open_image(options.forest_data_path)
    forest_gt = get_geotrans(options.forest_data_path)
    band_names = np.array(list(map(format_filename, forest_data.metadata['band names'])))
    # forest_labels = torch.from_numpy(forest_data.open_memmap())  # shape: 11996x12517x17
    forest_labels = forest_data.open_memmap()  # shape: 11996x12517x17

    hyper_labels = get_hyper_labels(hyper_image, forest_labels, hyper_gt, forest_gt)

    # -----add labels for 3 more tasks-----
    mainspecies = ['percentage_spruce', 'percentage_pine', 'percentage_broadleaf']
    additional_tasks = ['dbh_spruce', 'dbh_pine', 'dbh_birch']
    ids = []
    dbh_idx = np.where(band_names == 'dbh100_cm')[0]
    assert dbh_idx >= 0, 'Cannot find index of dbh'
    dbh_labels = hyper_labels[:, :, dbh_idx]
    for species in mainspecies:
        ids.append(np.where(band_names == species)[0])
    for i, task in enumerate(additional_tasks):
        band_names = np.append(band_names, task)
        assert ids[i] >= 0, 'Cannot find index of %s' % species
        task_labels = hyper_labels[:, :, ids[i]]
        new_task = dbh_labels * task_labels
        hyper_labels = np.concatenate((hyper_labels, new_task), axis=-1)
    # ----------

    # Current categorical classes: fertilityclass (0), soiltype (1), developmentclass (2), maintreespecies (9)
    cls_label_names = band_names[options.categorical_bands]
    reg_task_indices = np.array(range(hyper_labels.shape[-1]))
    reg_task_indices = np.delete(reg_task_indices, np.append(options.categorical_bands, options.ignored_bands))
    reg_label_names = band_names[reg_task_indices]
    n_reg_tasks = len(reg_task_indices)
    sum_pixels = np.sum(hyper_image, axis=-1)
    zero_count = R * C - np.count_nonzero(sum_pixels)
    print('Zero count in image:', R, C, B, sum_pixels.shape, zero_count)
    hyper_labels[sum_pixels == 0] = 0

    cls_labels = hyper_labels[:, :, options.categorical_bands]

    # Disable human data for now as there are only 19 Titta points in the map
    # hyper_labels = apply_human_data(options.human_data_path, hyper_labels, hyper_gt, band_names)
    hyper_labels, metadata = process_labels(hyper_labels, options.categorical_bands, options.ignored_bands,
                                            cls_label_names, reg_label_names, zero_count, save_path, options)

    print('Label visualization for classification tasks')
    visualize_label(cls_labels, cls_label_names, save_path)
    print('Label visualization for regression tasks')
    visualize_label(hyper_labels[:, :, -n_reg_tasks:], reg_label_names, save_path)

    image_norm_name = '%s/image_norm_%s.pt' % (save_path, options.normalize_method)
    tgt_name = '%s/%s.pt' % (save_path, options.tgt_file_name)
    metadata_name = '%s/%s.pt' % (save_path, options.metadata_file_name)
    src_name = '%s/%s.pt' % (save_path, options.src_file_name)

    metadata['cls_label_names'] = cls_label_names
    metadata['reg_label_names'] = reg_label_names
    metadata['ignore_zero_labels'] = options.ignore_zero_labels
    print('Metadata values: ', metadata)
    torch.save(metadata, metadata_name)
    torch.save(torch.from_numpy(hyper_labels), tgt_name)
    torch.save(torch.from_numpy(hyper_image), src_name)
    print('Target file has shapes {}'.format(hyper_labels.shape))
    del forest_labels, forest_data

    wavelength = np.array(hyper_data.metadata['wavelength'], dtype=float)

    save_as_rgb(hyper_image, wavelength, save_path)
    # storing L2 norm of the image based on normalization method
    if options.normalize_method == 'l2norm_along_channel':  # l2 norm along *band* axis
        # norm = np.linalg.norm(hyper_image, axis=2)
        norm_inv = np.zeros((R, C))
        for i in range(0, R):
            norm_inv[i] = np.linalg.norm(hyper_image[i, :, :], axis=1)
    elif options.normalize_method == 'l2norm_channel_wise':  # l2 norm separately for each channel
        norm_inv = np.linalg.norm(hyper_image, axis=(0, 1))

    norm_inv[norm_inv > 0] = 1.0 / norm_inv[norm_inv > 0]  # invert positive values
    norm_inv = torch.from_numpy(norm_inv)
    torch.save(norm_inv, image_norm_name)
    print('Source file has shapes {}'.format(norm_inv.shape))
    print('Processed files are stored under "./data" directory')
    print('End processing data...')


if __name__ == "__main__":
    main()
