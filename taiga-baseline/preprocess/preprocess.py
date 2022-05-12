#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Pre-process hyperspectral data
"""
import argparse
import os
import sys
import json
import re
import pandas as pd
import numpy as np
import spectral
import torch
import matplotlib.pyplot as plt

from utils import hyp2for, world2envi_p, save_as_rgb, visualize_label, format_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from tools.hypdatatools_img import get_geotrans

sys.stdout.flush()

#datadir = '/proj/deepsat/hyperspectral'
#datadir = '/scratch/project_2001284/hyperspectral'

def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir',
                        required=False, type=str,
                        default='.',
                        help='Path to input data directory')
    parser.add_argument('--hyperspec',
                        required=False, type=str,
                        default='<data_dir>/20170615_reflectance_mosaic_128b.hdr',
                        help='Path to hyperspectral data')
    parser.add_argument('--hyperspec_bands',
                        required=False, type=str,
                        default='0:110',
                        help='Range of hyperspectral bands to use')
    parser.add_argument('--forestdata',
                        required=False, type=str,
                        default='<data_dir>/forestdata.hdr',
                        help='Path to forest data')
    parser.add_argument('--human_data_path',
                        required=False, type=str,
                        default='<data_dir>/Titta2013.txt',
                        help='Path to human verified data (Titta points)')
    parser.add_argument('--save_dir',
                        required=False, type=str,
                        default='.',
                        help='Path to directory to save all generated files')
    parser.add_argument('--src_file_name',
                        required=False, type=str,
                        default='<save_dir>/hyperspectral_src',
                        help='Save hyperspectral image with this name')
    parser.add_argument('--tgt_file_name',
                        required=False, type=str,
                        default='<save_dir>/hyperspectral_tgt',
                        help='Save hyperspectral labels with this name')
    parser.add_argument('--metadata_file_name',
                        required=False, type=str,
                        default='<save_dir>/metadata',
                        help='Save metadata of the processed data with this name')
    parser.add_argument('--normalize_method',
                        type=str,
                        choices=['l2norm_along_channel', 'l2norm_channel_wise'],
                        default='l2norm_along_channel',
                        help='Normalization method for input image')
    parser.add_argument('--label_normalize_method',
                        type=str,
                        choices=['minmax_scaling', 'clip'],  # TODO: support z-score
                        default='clip',
                        help='Normalization method for target labels')
    parser.add_argument('--categorical_bands', type=str, nargs='+',
                        required=False, default=[],
                        help='List of categorical variables, defaults to all')
    parser.add_argument('--ignored_bands', nargs='+',
                        required=False, default=[],
                        help='List of band indices in the target labels to ignore')
    parser.add_argument('--ignore_zero_labels',
                        default=False, action='store_true',
                        help='Whether to ignore target labels with 0 values in data statistics')
    parser.add_argument('--remove_bad_data', default=False, action='store_true',
                        help='Remove bad data from forest labels')
    opt = parser.parse_args()
    opt.ignored_bands = list(map(int, opt.ignored_bands))
    return opt


def get_hyper_labels(data_dir, hyper_image, forest_labels,
                     hyper_gt, forest_gt, should_remove_bad_data):
    """
    Create hyperspectral labels from forest data
    :param hyper_image: W1xH1xC
    :param forest_labels: W2xH2xB
    :param hyper_gt
    :param forest_gt
    :param should_remove_bad_data
    :return:
    """
    rows, cols, _ = hyper_image.shape
    c, r = hyp2for((0, 0), hyper_gt, forest_gt)
    hyper_labels = np.array(forest_labels[r:(r+rows), c:(c+cols)])
    if should_remove_bad_data:
        stand_id_data    = spectral.open_image(data_dir+'/standids_in_pixels.hdr')
        stand_ids_full   = stand_id_data.open_memmap()
        stand_ids_mapped = np.array(stand_ids_full[r:(r + rows), c:(c + cols)], dtype='int')  # shape RxCx1
        stand_ids_mapped = np.squeeze(stand_ids_mapped)  # remove the single-dimensional entries, size RxC
        bad_stand_df     = pd.read_csv(data_dir+'/bad_stands.csv')
        bad_stand_list   = bad_stand_df['standid'].tolist()
        for stand_id in bad_stand_list:
            hyper_labels[stand_ids_mapped == stand_id] = 0
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


def handle_class_balancing(percentage, unique_values, threshold=5, remove_minor_classes=True):
    minor_percentages = percentage[percentage < threshold]
    remaining_percentages = percentage[percentage >= threshold]
    minor_classes = unique_values[percentage < threshold]
    remaining_classes = unique_values[percentage >= threshold]
    # 2 approaches to handle class imbalance
    #   1: remove all minor classes from data
    #   2: combine minor classes into a NEW common class (if minor_sum >= threshold)
    #   3: combine minor classes into smallest class above the threshold (otherwise)
    minor_sum = minor_percentages.sum()

    if remove_minor_classes:
        combined_class_idx = -1
        remaining_percentages = remaining_percentages / (100 - minor_sum)
    elif minor_sum >= threshold:
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

    print('    removed classes: {}, percentages: {}'.format(minor_classes, minor_percentages))
    print('    final classes: {}, percentages: {}'.format(remaining_classes, remaining_percentages))
    print()
    #print('Index dict:', index_dict)
    return index_dict, remaining_classes, remaining_percentages


def process_labels(labels, categorical_bands, ignored_bands, cls_label_names, reg_label_names, zero_count, save_path, options, forest_bands):
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
    #print(axs)
    
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
        print('Band {} ({}):'.format(b, cls_label_names[i]))
        print('    unique values = {}'.format(unique_values))
        print('    original distribution = {}'.format(" ".join(map("{:.2f}%".format, percentage))))
        print('    frequency = {}'.format(unique_counts))

        # plot_chart(axs[i // 2, i % 2], label_names[i], unique_values, percentage)  # original distribution
        # index_dict = dict([(val, idx) for (idx, val) in enumerate(unique_values)])
        
        threshold = 5
        index_dict, new_unique_values, new_percentages = handle_class_balancing(percentage, unique_values, threshold)
        plot_chart(axs[i // 2, i % 2], cls_label_names[i], new_unique_values, new_percentages)
        categorical[b] = new_unique_values
        # categorical_item = {}
        # categorical_item['id'] = i
        # categorical_item['name'] = cls_label_names[i]
        # categorical_item['values'] = forest_bands[cls_label_names[i]]['values']
        # categorical_item['original_values'] = unique_values.tolist()
        # index_new_unique_values = [{ 'index': idx , 'value': int(val) } for (idx, val) in enumerate(unique_values)]
        # categorical_item['new_unique_values'] = index_new_unique_values
        # categorical.append(categorical_item)

        one_hot = np.zeros((R, C, len(new_unique_values)), dtype=int)
        for row in range(R):
            for col in range(C):
                if band[row, col] == 0 and options.ignore_zero_labels:
                    continue
                idx = index_dict[band[row, col]]  # get the index of the value  band[row, col] in one-hot vector
                if idx > -1:
                    one_hot[row, col, idx] = 1
                else:
                    labels[row, col] = 0  # remove pixels containing the minor class

        if transformed_data is None:
            transformed_data = one_hot
        else:
            transformed_data = np.concatenate((transformed_data, one_hot), axis=2)
        num_classes += len(new_unique_values)

    # delete all the categorical classes from label data
    # labels = np.delete(labels, np.append(categorical_bands, ignored_bands), axis=2)
    labels = np.delete(labels, categorical_bands, axis=2)

    fig.savefig('%s/class_distribution.png' % save_path)
    plt.close(fig)

    # normalize data for regression task
    normalized_labels = []
    reg_stats = {}
    # reg_stats = []
    for b in range(labels.shape[2]):
        band = labels[:, :, b]
        true_max = np.max(band)
        if options.label_normalize_method == 'minmax_scaling':
            max_val = true_max
            min_val = np.min(band)
        elif options.label_normalize_method == 'clip':
            value_matrix = band[band > 0] if options.ignore_zero_labels else band
            # use 2 and 98 percentile as thresholding values
            # TODO: clip values can be passed as arguments
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

        cls_num = len(categorical)
        if reg_label_names[b] != "standID":
            print('Band {} ({}): lower bound = {}, upper bound = {}, true_max = {}'
                  .format(b + cls_num , reg_label_names[b], min_val, max_val, true_max))
        else:
            print('Band {} ({}): min = {}, max = {}'.format(b + cls_num, reg_label_names[b], min_val, true_max))
        print()
        reg_stats[b] = {'min': min_val, 'max': max_val, 'true_max': true_max}
        if max_val != min_val:
            band = (band - min_val) / (max_val - min_val)
        elif max_val != 0:  # if all items have the same non-zero value
            band.fill(0.5)
        else:  # if all are 0, if this happens, consider remove the whole band from data
            band.fill(0.0)
            print('Band with index %d has all zero values, consider removing it!' % b)

        # regression_item = {}
        # regression_item['id'] = b
        # regression_item['name'] = reg_label_names[b]
        # regression_item['min'] = int(min_val)
        # regression_item['max'] = int(max_val)
        # regression_item['true_max'] = int(true_max)
        # regression_item['scaling_factor'] = int(max_val - min_val)
        # # regression_item['range'] = forest_bands[cls_label_names[i]]['range']
        # # regression_item['unit'] = forest_bands[reg_label_names[b]]['unit'][1]
        # # forest_bands[cls_label_names[i]]['unit'][1]
        # reg_stats.append(regression_item)

        normalized_labels.append(band)
    normalized_labels = np.stack(normalized_labels, axis=2)
    # concatenate with newly transformed data
    normalized_labels = np.concatenate((transformed_data, normalized_labels), axis=2)

    metadata['categorical'] = categorical
    metadata['num_classes'] = num_classes
    metadata['regression'] = reg_stats

    return normalized_labels, metadata


def main():
    """ Main function to read input files and produce output files. """

    print('Starting to preprocess TAIGA hyperspectral and forest data...')
    options = parse_args()
    # print(options)

    # only take the first 110 spectral bands, the rest are noisy
    hyperspec_bands = options.hyperspec_bands
    m = re.match('^(\d+):(\d+)$', hyperspec_bands)
    assert m, 'hyperspec_bands value should be like 0:120'
    hyperspec_bands = (int(m.group(1)), int(m.group(2)))

    data_dir = options.data_dir
    save_dir = options.save_dir
    if save_dir[0]!='/' and save_dir[0]!='.':
        save_dir = os.getcwd()+'/'+save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hyperspec = options.hyperspec
    if hyperspec[0]!='/' and hyperspec[0]!='.':
        hyperspec = data_dir+'/'+hyperspec

    print('  data_dir         =', data_dir)
    print('  save_dir         =', save_dir)
    print('  hyperspec        =', hyperspec)
    print('  hyperspec_bands  =', *hyperspec_bands)
        
    hyper_data  = spectral.open_image(hyperspec)
    hyper_gt    = get_geotrans(hyperspec)
    hyper_image = hyper_data.open_memmap()
    hyper_image = np.array(hyper_image[:, :, hyperspec_bands[0]:
                                                hyperspec_bands[1]])  
    R, C, B = hyper_image.shape
    print('  hyperspec.shape  =', R, C, B);

    forestdata = options.forestdata
    if forestdata[0]!='/' and forestdata[0]!='.':
        forestdata = data_dir+'/'+forestdata
    print('  forestdata       =', forestdata)

    forest_data    = spectral.open_image(forestdata)
    forest_bands   = json.loads('{'+forest_data.metadata['band descriptions']+'}');
    forest_gt      = get_geotrans(forestdata)
    band_names_raw = np.array(forest_data.metadata['band names'])
    band_names     = np.array(list(map(format_filename, band_names_raw)))
    forest_labels  = forest_data.open_memmap()
    print('  forestdata.shape =', *forest_labels.shape);
    print()

    hyper_labels = get_hyper_labels(data_dir, hyper_image, forest_labels,
                                    hyper_gt, forest_gt, options.remove_bad_data)

    # -----add labels for 3 more tasks-----

    # TODO: change to dynamic
    mainspecies = ['percentage_of_spruce_', 'percentage_of_pine_', 'percentage_of_birch_']
    additional_tasks = ['dbh_spruce', 'dbh_pine', 'dbh_birch']
    ids = []
    dbh_idx = np.where(band_names == 'mean_dbh100_cm')[0]
    assert dbh_idx >= 0, 'Cannot find index of dbh'
    dbh_labels = hyper_labels[:, :, dbh_idx]
    for species in mainspecies:
        ids.append(np.where(band_names == species)[0])
    for i, task in enumerate(additional_tasks):
        band_names = np.append(band_names, task)

        band_names_raw = np.append(band_names_raw, task)

        assert ids[i] >= 0, 'Cannot find index of %s' % species
        task_labels = hyper_labels[:, :, ids[i]]
        new_task = dbh_labels * task_labels
        hyper_labels = np.concatenate((hyper_labels, new_task), axis=-1)
    # ----------
    # Categorical
    categorical_band_idxs = []
    if options.categorical_bands:
        for b in options.categorical_bands:
            idx = np.asarray(band_names_raw==b).nonzero()[0]
            assert idx>=0, 'Band "'+b+'" not in band names'
            assert b in forest_bands, 'Band "'+b+'" not in band descriptions'
            d = forest_bands[b]
            assert d['type']=='categorical', 'Band "'+b+'" is not categorical'
            categorical_band_idxs.append(idx[0])
    else:
        for b, d in forest_bands.items():
            idx = np.asarray(band_names_raw==b).nonzero()[0]
            assert idx>=0, 'Band "'+b+'" in band descriptions but not in names'
            if d['type']=='categorical':
                categorical_band_idxs.append(idx[0])

    cls_label_names = band_names[categorical_band_idxs]
    print('  categorical variables: (total '+str(len(categorical_band_idxs))+')\n')
    for i in range(len(categorical_band_idxs)):
        name = cls_label_names[i]
        desc = forest_bands[name]
        print('    {:2} {} "{}"'.format(categorical_band_idxs[i], name, desc['name']))
        for j, k in desc['values'].items():
            print('      {:2} : {}'.format(j, k))
        print()
            
    reg_task_indices = np.array(range(hyper_labels.shape[-1]))
    # reg_task_indices = np.delete(reg_task_indices, np.append(categorical_band_idxs,
    #                                                             options.ignored_bands))
    reg_task_indices = np.delete(reg_task_indices, categorical_band_idxs)
    reg_label_names = band_names[reg_task_indices]
    n_reg_tasks = len(reg_task_indices)
    sum_pixels = np.sum(hyper_image, axis=-1)
    zero_count = R * C - np.count_nonzero(sum_pixels)
    hyper_labels[sum_pixels == 0] = 0

    print('  regression variables: (total '+str(n_reg_tasks)+')\n')

    reg_label_names_raw = band_names_raw[reg_task_indices]
    for i in range(n_reg_tasks):
        name_raw = reg_label_names_raw[i]
        name_formatted = reg_label_names[i]
        labels = hyper_labels[:, :, reg_task_indices[i]]
        nonzero_label = labels[labels > 0]

        if name_raw == "standID":
            print('    {:2} {} {} [stand index]'.format(reg_task_indices[i], name_raw, name_formatted))
            keys = ["Min", "Max"]
            values = [np.amin(nonzero_label), np.amax(nonzero_label)]
            for k, v in zip(keys, values):
                print('      {} {}'.format(k, v))
            print()
            continue

        if "dbh_" in name_raw:
            print('    {:2} {} {} [derived variable]'.format(reg_task_indices[i], name_raw, name_formatted))
        else:
            print('    {:2} {} {}'.format(reg_task_indices[i], name_raw, name_formatted))
        keys = ["Min", "Max", "Mean", "Std"]
        values = [np.amin(nonzero_label), np.amax(nonzero_label), np.mean(nonzero_label), np.std(nonzero_label)]
        for k, v in zip(keys, values):
            print('      {} {}'.format(k, v))
        print()

    print('Zero count in hyperspectral data:', R, C, B, hyper_labels.shape, sum_pixels.shape, zero_count)
    print()

    cls_labels = hyper_labels[:, :, categorical_band_idxs]

    # Disable human data for now as there are only 19 Titta points in the map
    # hyper_labels = apply_human_data(options.human_data_path, hyper_labels, hyper_gt, band_names)
    hyper_labels, metadata = process_labels(hyper_labels, categorical_band_idxs, options.ignored_bands,
                                            cls_label_names, reg_label_names, zero_count, save_dir, options, forest_bands)

    print('Label visualization for classification tasks')
    visualize_label(cls_labels, cls_label_names, save_dir)
    print()
    print('Label visualization for regression tasks')
    visualize_label(hyper_labels[:, :, -n_reg_tasks:], reg_label_names, save_dir)
    print()

    image_norm_name = '%s/image_norm_%s.pt' % (save_dir, options.normalize_method)
    tgt_name = '%s/%s.pt' % (save_dir, options.tgt_file_name)
    metadata_name = '%s/%s.pt' % (save_dir, options.metadata_file_name)
    src_name = '%s/%s.pt' % (save_dir, options.src_file_name)
    saved_file_names = [image_norm_name, tgt_name, metadata_name, src_name]

    metadata['cls_label_names'] = cls_label_names
    metadata['reg_label_names'] = reg_label_names
    metadata['ignore_zero_labels'] = options.ignore_zero_labels
    print('Metadata: ', metadata)
    print()
    # with open('metadata.json', 'w', encoding='utf-8') as f:
    #     json.dump(metadata, f, ensure_ascii=False, indent=4)
    torch.save(metadata, metadata_name)
    torch.save(torch.from_numpy(hyper_labels), tgt_name)
    torch.save(torch.from_numpy(hyper_image), src_name)
    print('Target file has shape {}'.format(hyper_labels.shape))
    print()
    del forest_labels, forest_data

    wavelength = np.array(hyper_data.metadata['wavelength'], dtype=float)

    save_as_rgb(hyper_image, wavelength, save_dir)
    print()
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
    #print('Source file has shapes {}'.format(norm_inv.shape))
    print('Processed files are stored under "../data" directory:')
    for name in saved_file_names:
        print('    %s' % name)
    print('End preprocessing data...')


if __name__ == "__main__":
    main()
