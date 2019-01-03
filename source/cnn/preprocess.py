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

from input.utils import hyp2for, world2envi_p, open_as_rgb

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
                        # default='/proj/deepsat/hyperspectral/subset_A_20170615_reflectance.hdr',
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
                        help="Normalization method for input image")
    opt = parser.parse_args()

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
    hyper_labels = forest_labels[r:(r+rows), c:(c+cols)]

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


def process_labels(labels):
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
    :param labels: shape (RxCxB)
    :return:
    """
    # TODO: get 'categorical_classes' from parameter
    # Current categorical classes: fertilityclass (0), soiltype (1), developmentclass (2), maintreespecies (9)
    categorical_classes = [0, 1, 2, 9]  # contains the indices of the categorical classes in forest data
    useless_bands = [3]
    transformed_data = None
    num_classes = 0
    metadata = {}
    categorical = {}
    R, C, _ = labels.shape

    for i in categorical_classes:
        band = labels[:, :, i]
        unique_values = np.unique(band)
        print('Band {}: {}'.format(i, unique_values))
        index_dict = dict([(val, idx) for (idx, val) in enumerate(unique_values)])
        categorical[i] = unique_values
        one_hot = np.zeros((R, C, len(unique_values)), dtype=int)
        for row in range(R):
            for col in range(C):
                idx = index_dict[band[row, col]]  # get the index of the value  band[row, col] in one-hot vector
                one_hot[row, col, idx] = 1

        if transformed_data is None:
            transformed_data = one_hot
        else:
            transformed_data = np.concatenate((transformed_data, one_hot), axis=2)
        num_classes += len(unique_values)

    # delete all the categorical classes from label data
    labels = np.delete(labels, np.append(categorical_classes, useless_bands), axis=2)

    # normalize data for regression task
    for i in range(labels.shape[2]):
        max = np.max(labels[:, :, i])
        min = np.min(labels[:, :, i])
        if max != min:
            labels[:, :, i] = (labels[:, :, i] - min) / (max - min)
        elif max != 0:  # if all items have the same non-zero value
            labels[:, :, i].fill(0.5)
        else:  # if all are 0, if this happens, consider remove the whole band from data
            labels[:, :, i].fill(0.0)
            print('Band with index %d has all zero values, consider removing it!' % i)

    # concatenate with newly transformed data
    labels = np.concatenate((transformed_data, labels), axis=2)

    metadata['categorical'] = categorical
    metadata['num_classes'] = num_classes
    return labels, metadata


def main():
    print('Start processing data...')
    #######
    options = parse_args()
    print(options)
    hyper_data = spectral.open_image(options.hyper_data_path)
    hyper_gt = get_geotrans(options.hyper_data_path)
    hyper_image = hyper_data.open_memmap()
    hyper_image = hyper_image[:, :, 0:110]  # only take the first 110 spectral bands, the rest are noisy

    forest_data = spectral.open_image(options.forest_data_path)
    forest_gt = get_geotrans(options.forest_data_path)
    forest_columns = forest_data.metadata['band names']
    # forest_labels = torch.from_numpy(forest_data.open_memmap())  # shape: 11996x12517x17
    forest_labels = forest_data.open_memmap()  # shape: 11996x12517x17

    hyper_labels = get_hyper_labels(hyper_image, forest_labels, hyper_gt, forest_gt)
    # Disable human data for now as there are only 19 Titta points in the map
    # hyper_labels = apply_human_data(options.human_data_path, hyper_labels, hyper_gt, forest_columns)
    hyper_labels, metadata = process_labels(hyper_labels)

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    src_name = './data/%s_%s.pt' % (options.src_file_name, options.normalize_method)
    tgt_name = './data/%s.pt' % options.tgt_file_name
    metadata_name = './data/%s.pt' % options.metadata_file_name

    torch.save(metadata, metadata_name)
    torch.save(torch.from_numpy(hyper_labels), tgt_name)
    torch.save(torch.from_numpy(hyper_image), './data/hyper_image.pt')
    print('Target file has shapes {}'.format(hyper_labels.shape))
    del hyper_labels, forest_labels, forest_data

    wavelength_list = hyper_data.metadata['wavelength']
    wavelength = np.array(wavelength_list, dtype=float)

    # open_as_rgb(hyper_image, wavelength, options)

    R, C, B = hyper_image.shape
    # storing L2 norm of the image based on normalization method
    if options.normalize_method == 'l2norm_along_channel':  # l2 norm along *band* axis
        # norm = np.linalg.norm(hyper_image, axis=2)
        norm = np.zeros((R, C))
        for i in range(0, R):
            norm[i] = np.linalg.norm(hyper_image[i, :, :], axis=1)
    elif options.normalize_method == 'l2norm_channel_wise':  # l2 norm separately for each channel
        norm = np.linalg.norm(hyper_image, axis=(0, 1))

    norm[norm > 0] = 1.0 / norm[norm > 0]  # invert positive values
    torch.save(torch.from_numpy(norm), src_name)

    print('Source file has shapes {}'.format(norm.shape))
    print('Processed files are stored under "./data" directory')
    print('End processing data...')


if __name__ == "__main__":
    main()
