#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Training
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import pprint

from input.data_loader import get_loader
from input.focal_loss import FocalLoss
from input.utils import split_data, compute_data_distribution, remove_ignored_tasks, get_device
from models.model import ChenModel, LeeModel, PhamModel, SharmaModel, HeModel, ModelTrain, PhamModel3layers, \
    PhamModel3layers2, PhamModel3layers3, PhamModel3layers4, PhamModel3layers5, PhamModel3layers6, PhamModel3layers7, \
    PhamModel3layers8, PhamModel3layers9, PhamModel3layers10
from trainer import Trainer


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hyper_data_path',
                        required=False, type=str,
                        default='./data/mosaic/hyperspectral_src.pt',
                        help='Path to hyperspectral data')
    parser.add_argument('-src_norm_multiplier',
                        required=False, type=str,
                        default='./data/mosaic/image_norm_l2norm_along_channel.pt',
                        help='Path to file containing inverted norm (along color channel) of the source image')
    parser.add_argument('-tgt_path',
                        required=False, type=str,
                        default='./data/mosaic/hyperspectral_tgt_normalized.pt',
                        help='Path to training labels')
    parser.add_argument('-input_normalize_method',
                        required=False, type=str,
                        default='l2_norm', choices=['l2_norm', 'minmax_scaling'],
                        help='Normalization method for input image')
    parser.add_argument('-metadata',
                        type=str,
                        default='./data/mosaic/metadata.pt',
                        help='Path to training metadata (generated during preprocessing stage)')
    parser.add_argument('-gpu',
                        type=int, default=-1,
                        help='Gpu id to be used, default is -1 which means cpu')
    parser.add_argument('-hyper_data_header',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/20170615_reflectance_mosaic_128b.hdr',
                        help='Path to hyperspectral data header')
    parser.add_argument('-data_split_path',
                        required=False, type=str,
                        default='./data/mosaic/splits',
                        help='Path to training and validation dataset')
    # Training options
    train = parser.add_argument_group('Training')
    train.add_argument('-epoch', type=int,
                       default=10,
                       help='Number of training epochs, default is 10')
    train.add_argument('-patch_size', type=int,
                       default=27,
                       help='Size of the spatial neighbourhood, default is 11')
    train.add_argument('-patch_stride', type=int,
                       default=1,
                       help='Number of pixels to skip for each image patch while sliding over the training image')
    train.add_argument('-lr', type=float,
                       default=1e-3,
                       help='Learning rate, default is 1e-3')
    train.add_argument('-batch_size', type=int,
                       default=64,
                       help='Batch size, default is 64')
    train.add_argument('-train_from', type=str,
                       default='',
                       help='Path to checkpoint to start training from.')
    train.add_argument('-model', type=str,
                       default='ChenModel', choices=['ChenModel', 'LeeModel', 'PhamModel', 'SharmaModel', 'HeModel',
                                                     'PhamModel3layers', 'PhamModel3layers2', 'PhamModel3layers3',
                                                     'PhamModel3layers4', 'PhamModel3layers5', 'PhamModel3layers6',
                                                     'PhamModel3layers7', 'PhamModel3layers8', 'PhamModel3layers9',
                                                     'PhamModel3layers10'],
                       help='Name of deep learning model to train with, options are [ChenModel | LeeModel]')
    train.add_argument('-save_dir', type=str,
                       default='',
                       help='Directory to save model. If not specified, use name of the model')
    train.add_argument('-report_frequency', type=int,
                       default=20,
                       help='Report training result every "report_frequency" steps')
    train.add_argument('-use_visdom', default=False, action='store_true',
                       help="Enable visdom to visualize training process")
    train.add_argument('-visdom_server', type=str,
                       default='http://localhost',
                       help='Default visdom server')
    train.add_argument('-loss_balancing', type=str, choices=['grad_norm', 'equal_weights', 'uncertainty'],
                       default='grad_norm',
                       help='Specify loss balancing method for multi-task learning')
    train.add_argument('-class_balancing', type=str, choices=['cost_sensitive', 'focal_loss', 'CRL'],
                       default='cost_sensitive',
                       help='Specify method to handle class imbalance. Available options: '
                            '[cost sensitive | class rectification loss]')
    train.add_argument('-ignored_cls_tasks', nargs='+',
                        required=False, default=[],
                        help='List of classification task indices to ignore, indexing starts from 0')
    train.add_argument('-ignored_reg_tasks', nargs='+',
                        required=False, default=[],
                        help='List of regression task indices to ignore, indexing starts from 0')
    train.add_argument('-augmentation', type=str, choices=['flip', 'radiation_noise', 'mixture_noise'],
                       default=None,
                       help='Specify augmentation method')
    train.add_argument('-keep_best', type=int,
                       default=10, help='Specify the number of best models to keep.')

    opt = parser.parse_args()
    opt.ignored_cls_tasks = list(map(int, opt.ignored_cls_tasks))
    opt.ignored_reg_tasks = list(map(int, opt.ignored_reg_tasks))

    return opt


def get_input_data(metadata_path):
    """
    Get info such as number of classes for categorical classes
    :return:
    """
    metadata = torch.load(metadata_path)

    return metadata


def main():
    print('Start training...')
    print('System info: ', sys.version)
    print('Numpy version: ', np.__version__)
    print('Torch version: ', torch.__version__)
    print()
    #######
    checkpoint = None
    options = parse_args()

    if options.train_from:
        print('Loading checkpoint from %s' % options.train_from)
        print('Overwrite some options with values from checkpoint!!!')
        checkpoint = torch.load(options.train_from)
        ckpt_options = checkpoint['options']
        options.patch_size = ckpt_options.patch_size
        options.patch_stride = ckpt_options.patch_stride
        options.model = ckpt_options.model

    # TODO: check for minimum patch_size
    #print('Training options: {}'.format(options))
    print('Training options:')
    pprint.pprint(vars(options))
    print()
    device = get_device(options.gpu)

    visualize = options.use_visdom
    visualizer = None
    if visualize:
        env_name = 'Train-{}'.format(options.save_dir)
        # 'server' option is needed because of this error: https://github.com/facebookresearch/visdom/issues/490
        visualizer = visdom.Visdom(server=options.visdom_server, env=env_name)
        if not visualizer.check_connection:
            print("Visdom server is unreachable. Run `bash server.sh` to start the server.")
            visualizer = None

    metadata = get_input_data(options.metadata)
    hyper_image = torch.load(options.hyper_data_path)
    hyper_labels = torch.load(options.tgt_path)
    norm_inv = torch.load(options.src_norm_multiplier).float()

    # use percentage of main tree species as mask if ignore_zero_labels is True
    # => only care about forest areas
    out_cls = metadata['num_classes']
    idx = np.where(metadata['reg_label_names'] == 'leaf_area_index100')[0]
    mask = hyper_labels[:, :, out_cls + idx] if metadata['ignore_zero_labels'] else norm_inv
    #print((mask != 0).sum().tolist())
    # remove ignored tasks
    hyper_labels_cls, hyper_labels_reg = remove_ignored_tasks(hyper_labels, options, metadata)
    categorical = metadata['categorical']
    print('Metadata values:')
    pprint.pprint(metadata)
    print()
    out_cls = metadata['num_classes']
    out_reg = hyper_labels_reg.shape[-1]

    options.no_classification = True if out_cls == 0 else False
    options.no_regression = True if out_reg == 0 else False

    # hyper_labels_cls = hyper_labels[:, :, :out_cls]
    # hyper_labels_reg = hyper_labels[:, :, out_cls:]

    R, C, num_bands = hyper_image.shape

    if os.path.isfile(options.data_split_path + '/train_set.npy') and os.path.isfile(options.data_split_path + '/val_set.npy'):
        #print('Loading data split...')
        print('Loading data split from ' + options.data_split_path)
        train_set = np.load(options.data_split_path + '/train_set.npy', allow_pickle=True)
        val_set = np.load(options.data_split_path + '/val_set.npy', allow_pickle=True)
        test_set = np.load(options.data_split_path + '/test_set.npy', allow_pickle=True)
        # origin_set = np.load(options.data_split_path + '/origin_set.npy', allow_pickle=True)
    else:
        train_set, test_set, val_set, origin_set = split_data(R, C, mask, hyper_labels_cls, hyper_labels_reg, options.patch_size, options.patch_stride)
        if not os.path.exists(options.data_split_path):
            os.makedirs(options.data_split_path)
        np.save(options.data_split_path + '/train_set.npy', train_set)
        np.save(options.data_split_path + '/val_set.npy', val_set)
        np.save(options.data_split_path + '/test_set.npy', test_set)
        np.save(options.data_split_path + '/origin_set.npy', origin_set)

    print('Data distribution on training set')
    class_weights = compute_data_distribution(hyper_labels_cls, train_set, categorical)
    print('Data distribution on validation set')
    _ = compute_data_distribution(hyper_labels_cls, val_set, categorical)

    # Model construction
    model_name = options.model

    reduction = 'sum' if options.loss_balancing == 'uncertainty' else 'mean'
    loss_reg = nn.MSELoss(reduction=reduction)
    # loss_reg = nn.MSELoss()
    loss_cls_list = []

    if options.class_balancing == 'cost_sensitive' or options.class_balancing == 'CRL':
        for i in range(len(categorical.keys())):
            loss_cls_list.append(nn.CrossEntropyLoss(weight=class_weights[i].to(device)))
    elif options.class_balancing == 'focal_loss':
        for i in range(len(categorical.keys())):
            # loss_cls_list.append(FocalLoss(class_num=len(class_weights[i]), alpha=torch.tensor(class_weights[i]), gamma=2))
            loss_cls_list.append(FocalLoss(balance_param=class_weights[i].to(device), weight=class_weights[i].to(device)))
    else:
        for i in range(len(categorical.keys())):
            loss_cls_list.append(nn.CrossEntropyLoss())

    if model_name == 'ChenModel':
        model = ChenModel(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel':
        model = PhamModel(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
        # loss_reg = nn.L1Loss()
    elif model_name == 'PhamModel3layers':
        model = PhamModel3layers(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers2':
        model = PhamModel3layers2(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers3':
        model = PhamModel3layers3(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers4':
        model = PhamModel3layers4(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers5':
        model = PhamModel3layers5(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers6':
        model = PhamModel3layers6(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers7':
        model = PhamModel3layers7(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers8':
        model = PhamModel3layers8(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers9':
        model = PhamModel3layers9(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'PhamModel3layers10':
        model = PhamModel3layers10(num_bands, out_reg, metadata, patch_size=options.patch_size, n_planes=32)
    elif model_name == 'SharmaModel':
        model = SharmaModel(num_bands, out_reg, metadata, patch_size=options.patch_size)
    elif model_name == 'HeModel':
        model = HeModel(num_bands, out_reg, metadata, patch_size=options.patch_size)
    elif model_name == 'LeeModel':
        model = LeeModel(num_bands, out_cls, out_reg)

    multiplier = None if options.input_normalize_method == 'minmax_scaling' else norm_inv

    train_loader = get_loader(hyper_image,
                              multiplier,
                              hyper_labels_cls,
                              hyper_labels_reg,
                              train_set,
                              options.batch_size,
                              model_name=model_name,
                              is_3d_convolution=True,
                              augmentation=options.augmentation,
                              patch_size=options.patch_size,
                              shuffle=True)
    val_loader = get_loader(hyper_image,
                            multiplier,
                            hyper_labels_cls,
                            hyper_labels_reg,
                            val_set,
                            options.batch_size,
                            model_name=model_name,
                            is_3d_convolution=True,
                            augmentation=None,
                            patch_size=options.patch_size,
                            shuffle=True)
    test_loader = get_loader(hyper_image,
                             multiplier,
                             hyper_labels_cls,
                             hyper_labels_reg,
                             test_set,
                             options.batch_size,
                             model_name=model_name,
                             is_3d_convolution=True,
                             augmentation=None,
                             patch_size=options.patch_size,
                             shuffle=True)
    # origin_loader = get_loader(hyper_image,
    #                             multiplier,
    #                             hyper_labels_cls,
    #                             hyper_labels_reg,
    #                             origin_set,
    #                             options.batch_size,
    #                             model_name=model_name,
    #                             is_3d_convolution=True,
    #                             augmentation=None,
    #                             patch_size=options.patch_size,
    #                             shuffle=False)

    print('Dataset sizes: train={}, val={}, test={} patches'.
            format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))

    model = model.to(device)
    # run model summary before converting model to DataParallel
    # as torch summary doesn't work with torch 1.0.0
    with torch.no_grad():
        print('Model summary: ')
        for input, _, _, _ in train_loader:
            break

        summary(model,
                input.shape[1:],
                batch_size=options.batch_size,
                device=device.type)

    if options.gpu > -1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    modelTrain = ModelTrain(model, loss_cls_list, loss_reg, metadata, options)

    # do this before defining the optimizer:  https://pytorch.org/docs/master/optim.html#constructing-it
    modelTrain = modelTrain.to(device)
    optimizer = optim.Adam(modelTrain.parameters(), lr=options.lr, weight_decay=0.0001)

    # End model construction

    if checkpoint is not None:
        modelTrain.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    print(modelTrain)
    print('Classification loss function:', loss_cls_list)
    print('Regression loss function:', loss_reg)
    print('Scheduler:', scheduler.__dict__)

    trainer = Trainer(modelTrain, optimizer, scheduler, device, visualizer, metadata,
                      options, hyper_labels_reg, checkpoint)
    trainer.train(train_loader, val_loader, test_loader)
    print('End training...')


if __name__ == "__main__":
    main()
