#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process hyperspectral data
"""
import argparse
import os

import numpy as np
import spectral
from sklearn.model_selection import train_test_split

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
    xy1 = envi2world_p(xy0, hgt)
    xy2 = world2envi_p(xy1, fgt)
    return int(np.floor(xy2[0] + 0.1)), int(np.floor(xy2[1] + 0.1))


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hyperspectral_path',
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
    opt = parser.parse_args()

    return opt


def get_image_patch(hyp_image, x, y, size):
    """
    Get an image patch from the center point (x,y) with dimension size x size
    :param hyp_image: hyperspectral image
    :param x: x-coordinate
    :param y: y-coordinate
    :param size: image size
    :return: image patch
    """
    # x = int(x.round())
    # y = int(y.round())
    x_left = x - size // 2
    x_right = x_left + size
    y_left = y - size // 2
    y_right = y_left + size
    image = hyp_image[x_left:x_right, y_left:y_right, :]
    return image


def augment_points(hyp_image, x, y, size):
    """
    Augment the input data. Notice the size should be within 70cm from the center (x,y)
    :param hyp_image: hyperspectral image
    :param x: x-coordinate of a Titta point in image coordinate
    :param y: y-coordinate of a Titta point in image coordinate
    :param size: size of image patch to return
    :return: a list of 9 points
    """

    x = int(round(x)) if type(x) is not 'int' else x
    y = int(round(y)) if type(y) is not 'int' else y

    patches = []
    operator = [-1, 0, 1]
    centers = []

    # TODO: replace with random crops
    for i in operator:
        for j in operator:
            centers.append((x + size*i, y + size*j))

    for x, y in centers:
        patches.append(get_image_patch(hyp_image, x, y, size))
    return patches


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


def main():
    dirlist = ['/proj/deepsat/hyperspectral',
               '~/airobest/hyperspectral',
               '.']

    for d in dirlist:
        hdir = os.path.expanduser(d)
        if os.path.isdir(hdir):
            break

    hyphdr = hdir + '/subset_A_20170615_reflectance.hdr'
    # hyphdr  = '../20170615_reflectance_mosaic_128b.hdr'
    hypdata = spectral.open_image(hyphdr)
    hypmap = hypdata.open_memmap()
    hypgt = get_geotrans(hyphdr)
    print('Spectral: {:d} x {:d} pixels, {:d} bands in {:s}'. \
          format(hypdata.shape[1], hypdata.shape[0], hypdata.shape[2], hyphdr))

    split_data(hypmap.shape[0], hypmap.shape[1])

    forhdr = hdir + '/forestdata.hdr'
    fordata = spectral.open_image(forhdr)
    formap = fordata.open_memmap()
    forgt = get_geotrans(forhdr)
    print('Forest:   {:d} x {:d} pixels, {:d} bands in {:s}'. \
          format(fordata.shape[1], fordata.shape[0], fordata.shape[2], forhdr),
          flush=True)

    titta_params = []
    titta_world = []  # stores indexes and coordinates of Titta points with world coordinates
    titta_xy = []  # stores Titta points with image coordinates
    titta_val = []  # stores other values of Titta points

    #######
    options = parse_args()
    print(options)
    hyp_data = spectral.open_image(options.hyperspectral_path)
    hyp_map = hyp_data.open_memmap()
    geo_transform = get_geotrans(options.hyperspectral_path)

    with open(options.human_data_path, encoding='utf-16') as tf:
        ln = 0
        for ll in tf:
            lx = ll.rstrip().split('\t')

            if ln == 0:
                titta_params = lx[3:]
            else:
                idxy = [int(lx[0]), int(lx[1]), int(lx[2])]
                titta_world.append(idxy)
                xy = world2envi_p(idxy[1:3], hypgt)
                titta_xy.append(xy)
                v = []
                for i in range(3, len(lx)):
                    vf = 0.0
                    if lx[i] != '#DIV/0!':
                        vf = float(lx[i])
                    v.append(vf)
                titta_val.append(v)

                patches = augment_points(hyp_data, xy[0], xy[1], 10)
                print(patches.shape)
            ln += 1

if __name__ == "__main__":
    main()
