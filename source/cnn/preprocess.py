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
from sklearn import preprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from tools.hypdatatools_img import get_geotrans

sys.stdout.flush()


# TODO: move to tools
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


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hyper_data_path',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/subset_A_20170615_reflectance.hdr',
                        # default='/proj/deepsat/hyperspectral/20170615_reflectance_mosaic_128b.hdr',
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
    # TODO: confirm if forest data and hyperspectral data
    # are spatially correspondent. If yes, no need to convert
    # all the pixels, only the first pixel is needed
    forest_rows, forest_cols, num_bands = forest_labels.shape
    rows, cols, _ = hyper_image.shape
    hyper_labels = np.zeros((rows, cols, num_bands))

    for row in range(rows):
        for col in range(cols):
            # get coordinate of (row, col) in forest map
            c, r = hyp2for((col, row), hyper_gt, forest_gt)
            # assert forest_cols >= c >= 0 and forest_rows >= r >= 0, \
            #     "Invalid coordinates after conversion: %s %s --> %s %s " % (col, row, c, r)
            hyper_labels[row, col] = forest_labels[r, c]

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
    metadata_name = './data/metadata.pt'

    # L2 normalization
    R, C, B = hyper_image.shape
    hyper_image = hyper_image.reshape(-1, B)  # flatten image
    if options.normalize_method == 'l2norm_along_channel':  # l2 normalize along *band* axis
        hyper_image = preprocessing.normalize(hyper_image, norm='l2', axis=1)
    elif options.normalize_method == 'l2norm_channel_wise':  # l2 normalize separately for each channel
        hyper_image = preprocessing.normalize(hyper_image, norm='l2', axis=0)
    # np.linalg.norm(hyper_image[0,:]) should be 1.0
    hyper_image = hyper_image.reshape(R, C, B)  # reshape to original size

    torch.save(metadata, metadata_name)
    torch.save(torch.from_numpy(hyper_image), src_name)
    torch.save(torch.from_numpy(hyper_labels), tgt_name)

    print('Source and target files have shapes {}, {}'.format(hyper_image.shape, hyper_labels.shape))
    print('Processed files are stored under "./data" directory')
    print('End processing data...')


if __name__ == "__main__":
    main()
