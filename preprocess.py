#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process hyperspectral data
"""
import argparse
import os

import pandas as pd
import numpy as np
import spectral
from sklearn.model_selection import train_test_split
import torch

from tools.hypdatatools_img import get_geotrans


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
    opt = parser.parse_args()

    return opt


def split_data(rows, cols):
    coords = []
    for i in range(rows):
        for j in range(cols):
            coords.append((i, j))

    train, test = train_test_split(coords, train_size=0.8, random_state=123, shuffle=True)
    train, val = train_test_split(train, train_size=0.9, random_state=123, shuffle=True)
    np.savetxt('train.txt', train, fmt="%d")
    np.savetxt('test.txt', test, fmt="%d")
    np.savetxt('val.txt', val, fmt="%d")


def get_hyper_labels(hyper_image, forest_labels, hyper_gt, forest_gt):
    """
    Create hyperspectral labels from forest data
    :param hyper_image: W1xH1xC
    :param forest_labels: W2xH2xB
    :return:
    """
    num_bands = forest_labels.shape[2]
    rows, cols = hyper_image.shape
    hyper_labels = torch.zeros(rows, cols, num_bands)

    for row in range(rows):
        for col in range(cols):
            # get coordinate of (row, col) in forest map
            c, r = hyp2for((col, row), hyper_gt, forest_gt)
            assert cols >= c >= 0 and rows >= r >= 0, \
                "Invalid coordinates after conversion: %s %s --> %s %s " % (col, row, c, r)
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
    coords = data[1:3]
    labels = data[3:]

    titta_columns = df.columns.values[3:]
    print('Column mappings from Titta to Forest data')
    for titta_idx, forest_idx in column_mappings.items():
        print('%s --> %s ' % (titta_columns[titta_idx], forest_columns[forest_idx]))

    for idx, x, y in enumerate(coords):
        c, r = world2envi_p((x, y), hyper_gt)

        if not in_hypmap(hyper_labels.shape[1], hyper_labels.shape[0], c, r):
            continue

        label = labels[idx]
        for titta_idx, forest_idx in column_mappings.items():
            hyper_labels[r, c, forest_idx] = label[titta_idx]

    return hyper_labels


def main():

    #######
    options = parse_args()
    print(options)
    hyper_data = spectral.open_image(options.hyper_data_path)
    hyper_image = hyper_data.open_memmap()
    hyper_gt = get_geotrans(options.hyper_data_path)

    forest_data = spectral.open_image(options.forest_data_path)
    forest_labels = forest_data.open_memmap()
    forest_gt = get_geotrans(options.forest_data_path)
    forest_columns = forest_data.metadata['band names']

    split_data(hyper_image.shape[0], hyper_image.shape[1])

    hyper_labels = get_hyper_labels(hyper_image, forest_labels, hyper_gt, forest_gt)
    hyper_labels = apply_human_data(options.human_data_path, hyper_labels, hyper_gt, forest_columns)

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    src_name = './data/%s.pt' % options['src_file_name']
    tgt_name = './data/%s.pt' % options['tgt_file_name']

    torch.save(hyper_image, src_name)
    torch.save(hyper_labels, tgt_name)


if __name__ == "__main__":
    main()
