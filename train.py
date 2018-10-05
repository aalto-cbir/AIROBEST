#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Training
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from models.model import ChenModel
from input.utils import split_data
from input.data_loader import get_loader


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_path',
                        required=False, type=str,
                        default='./data/hyperspectral_src.pt',
                        help='Path to training input file')
    parser.add_argument('-tgt_path',
                        required=False, type=str,
                        default='./data/hyperspectral_tgt.pt',
                        help='Path to training labels')
    parser.add_argument('-gpu',
                        type=int, default=-1,
                        help="Gpu id to be used, default is -1 which means cpu")
    # Training options
    train = parser.add_argument_group('Training')
    train.add_argument('-epoch', type=int,
                       default=10,
                       help="Number of training epochs, default is 10")
    train.add_argument('-patch_size', type=int,
                       default=27,
                       help="Size of the spatial neighbourhood, default is 11")
    train.add_argument('-lr', type=float,
                       default=1e-3,
                       help="Learning rate, default is 1e-3")
    train.add_argument('-batch_size', type=int,
                       default=64,
                       help="Batch size, default is 64")
    opt = parser.parse_args()

    return opt


def get_input_data(metadata_path):
    """
    Get info such as number of classes for categorical classes
    :return:
    """
    metadata = torch.load(metadata_path)

    return metadata


def get_device(id):
    device = torch.device('cpu')
    if id > -1 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(id))
    print("Number of GPUs available %i" % torch.cuda.device_count())
    print("Training on device: %s" % device)
    return device


def save_checkpoint(model, model_name, epoch):
    path = './checkpoint/{}'.format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), '{}/{}_{}.pt'.format(path, model_name, epoch))
    print('Saving model at epoch %d' % epoch)


def validate(net, val_loader, device):
    # TODO: fix me
    pass


def train(net, optimizer, loss_fn, train_loader, val_loader, device, options):
    """
    Training

    TODO: checkpoint
    :param net:
    :param optimizer:
    :param loss:
    :param train_loader:
    :param val_loader:
    :param options:
    :return:
    """
    device = device
    epoch = options.epoch
    save_every = 1  # specify number of epochs to save model
    train_step = 0

    net.to(device)

    losses = np.array([])

    for e in range(epoch):
        net.train()  # TODO: check docs

        for idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            predict = net(src)
            loss = loss_fn(predict, tgt)

            loss.backward()

            optimizer.step()

            if train_step % 200 == 0:
                print('Traing loss {}'.format(loss.item()))

            np.append(losses, loss.item())
            train_step += 1

        # TODO: validation
        if val_loader is not None:
            validate(net, val_loader, device)

        if e % save_every == 0:
            save_checkpoint(net, 'Chen-model', e)


def main():
    print('Start training...')
    print('System info: ', sys.version)
    print('Numpy version: ', np.__version__)
    print('Torch version: ', torch.__version__)
    #######
    options = parse_args()
    device = get_device(options.gpu)
    # device = torch.device('cuda:0')
    # TODO: check for minimum patch_size
    print('Training options: {}'.format(options))

    metadata = get_input_data('./data/metadata.pt')
    output_classes = metadata['num_classes']
    assert output_classes > 0, 'Number of classes has to be > 0'

    hyper_image = torch.load(options.src_path).float()
    hyper_labels = torch.load(options.tgt_path)
    # TODO: only need labels for classification task for now
    hyper_labels_cls = hyper_labels[:, :, :output_classes]
    hyper_labels_reg = hyper_labels[:, :, (output_classes+1):]

    # maybe only copy to gpu during computation?
    hyper_image.to(device)
    # hyper_labels.to(device)
    hyper_labels_cls.to(device, dtype=torch.float32)
    hyper_labels_reg.to(device, dtype=torch.float32)

    R, C, B = hyper_image.shape
    # TODO: Convert image representation, should be done in preprocessing stage
    # hyper_image = hyper_image.permute(2, 0, 1)

    train_set, test_set, val_set = split_data(R, C, options.patch_size)

    train_loader = get_loader(hyper_image,
                              hyper_labels_cls,
                              train_set,
                              options.batch_size,
                              patch_size=options.patch_size,
                              shuffle=True)
    val_loader = get_loader(hyper_image,
                            hyper_labels_cls,
                            val_set,
                            options.batch_size,
                            patch_size=options.patch_size,
                            shuffle=True)

    # Model construction
    W, H, num_bands = hyper_image.shape
    model = ChenModel(num_bands, output_classes)
    optimizer = optim.Adam(model.parameters(), lr=options.lr)

    # loss = nn.CrossEntropyLoss()  # doesn't work for multi-target
    loss = nn.MultiLabelSoftMarginLoss(size_average=False)
    # End model construction

    train(model, optimizer, loss, train_loader, val_loader, device, options)
    print('End training...')


if __name__ == "__main__":
    main()
