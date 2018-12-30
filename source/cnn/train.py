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
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import visdom

from models.model import ChenModel, LeeModel, PhamModel
from input.utils import split_data
from input.data_loader import get_loader
from trainer import Trainer


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hyper_data_path',
                        required=False, type=str,
                        default='/proj/deepsat/hyperspectral/subset_A_20170615_reflectance.hdr',
                        help='Path to hyperspectral data')
    parser.add_argument('-src_norm_multiplier',
                        required=False, type=str,
                        default='../../data/hyperspectral_src_l2norm.pt',
                        help='Path to file containing inverted norm (along color channel) of the source image')
    parser.add_argument('-tgt_path',
                        required=False, type=str,
                        default='../../data/hyperspectral_tgt.pt',
                        help='Path to training labels')
    parser.add_argument('-metadata',
                        type=str,
                        default='../../data/metadata.pt',
                        help="Path to training metadata (generated during preprocessing stage)")
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
    train.add_argument('-patch_stride', type=int,
                       default=1,
                       help="Number of pixels to skip for each image patch while sliding over the training image")
    train.add_argument('-lr', type=float,
                       default=1e-3,
                       help="Learning rate, default is 1e-3")
    train.add_argument('-batch_size', type=int,
                       default=64,
                       help="Batch size, default is 64")
    train.add_argument('-train_from', type=str,
                       default='',
                       help="Path to checkpoint to start training from.")
    train.add_argument('-model', type=str,
                       default='ChenModel', choices=['ChenModel', 'LeeModel'],
                       help="Name of deep learning model to train with, options are [ChenModel | LeeModel]")
    train.add_argument('-save_dir', type=str,
                       default='',
                       help="Directory to save model. If not specified, use name of the model")
    train.add_argument('-report_frequency', type=int,
                       default=20,
                       help="Report training result every 'report_frequency' steps")
    train.add_argument('-use_visdom', type=bool,
                       default=True,
                       help="Enable visdom to visualize training process")
    train.add_argument('-visdom_server', type=str,
                       default='http://localhost',
                       help="Default visdom server")
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


def save_checkpoint(model, optimizer, model_name, epoch, options):
    """
    Saving model's state dict
    :param model: model to save
    :param optimizer: optimizer to save
    :param model_name: model will be saved under this name
    :param epoch: the epoch when model is saved
    :return:
    """
    path = './checkpoint/{}'.format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'options': options
    }
    torch.save(state, '{}/{}_{}.pt'.format(path, model_name, epoch))
    print('Saving model at epoch %d' % epoch)


def compute_accuracy(predict, tgt, metadata):
    """
    Return number of correct prediction of each tgt label
    :param predict:
    :param tgt:
    :param metadata:
    :return:
    """
    n_correct = 0  # vector or scalar?

    # reshape tensor in (*, n_cls) format
    # this is mainly for LeeModel that output the prediction for all pixels
    # from the source image with shape (batch, patch, patch, n_cls)
    n_cls = tgt.shape[-1]
    predict = predict.view(-1, n_cls)
    tgt = tgt.view(-1, n_cls)
    #####

    categorical = metadata['categorical']
    num_classes = 0
    for idx, values in categorical.items():
        count = len(values)
        pred_class = predict[:, num_classes:(num_classes + count)]
        tgt_class = tgt[:, num_classes:(num_classes + count)]
        pred_indices = pred_class.argmax(-1)  # get indices of max values in each row
        tgt_indices = tgt_class.argmax(-1)
        true_positive = torch.sum(pred_indices == tgt_indices).item()
        n_correct += true_positive
        num_classes += count

    # return n_correct divided by number of labels * batch_size
    return n_correct / (len(predict) * len(categorical.keys()))


def validate(net, criterion_cls, criterion_reg, val_loader, device, metadata):
    sum_loss = 0
    N_samples = 0
    sum_accuracy = 0
    for idx, (src, tgt_cls, tgt_reg) in enumerate(val_loader):
        src = src.to(device, dtype=torch.float32)
        tgt_cls = tgt_cls.to(device, dtype=torch.float32)
        tgt_reg = tgt_reg.to(device, dtype=torch.float32)
        N_samples += len(src)

        with torch.no_grad():
            pred_cls, pred_reg = net(src)
            loss_cls = criterion_cls(pred_cls, tgt_cls)
            loss_reg = criterion_reg(pred_reg, tgt_reg)
            loss = 1 * loss_cls + 3 * loss_reg

            sum_loss += loss.item()
            sum_accuracy += compute_accuracy(pred_cls, tgt_cls, metadata)

    # return average validation loss
    average_loss = sum_loss / len(val_loader)
    # accuracy = n_correct * 100 / N_samples
    accuracy = sum_accuracy * 100 / len(val_loader)
    return average_loss, accuracy


def train(net, optimizer, criterion_cls, criterion_reg, train_loader, val_loader, device, metadata,
          options, scheduler=None, visualize=None):
    """
    Training

    :param net:
    :param optimizer:
    :param criterion_cls:
    :param criterion_reg:
    :param train_loader:
    :param val_loader:
    :param device:
    :param metadata:
    :param options:
    :param scheduler:
    :param visualize:
    :return:
    """
    epoch = options.epoch
    start_epoch = options.start_epoch + 1 if 'start_epoch' in options else 1
    save_every = 1  # specify number of epochs to save model
    train_step = 0
    sum_loss = 0.0
    avg_losses = []
    val_losses = []
    val_accuracies = []
    loss_window = None

    net.to(device)

    losses = []

    print('Start training from epoch: ', start_epoch)
    for e in range(start_epoch, epoch + 1):
        net.train()
        epoch_loss = 0.0

        for idx, (src, tgt_cls, tgt_reg) in enumerate(train_loader):
            src = src.to(device, dtype=torch.float32)
            tgt_cls = tgt_cls.to(device, dtype=torch.float32)
            tgt_reg = tgt_reg.to(device, dtype=torch.float32)
            # tgt = tgt.to(device, dtype=torch.int64)

            optimizer.zero_grad()
            pred_cls, pred_reg = net(src)
            loss_cls = criterion_cls(pred_cls, tgt_cls)
            loss_reg = criterion_reg(pred_reg, tgt_reg)
            loss = 1 * loss_cls + 3 * loss_reg

            sum_loss += loss.item()
            epoch_loss += loss.item()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if train_step % options.report_frequency == 0:
                avg_losses.append(np.mean(losses[-100:]))
                print('Training loss at step {}: {:.5f}, average loss: {:.5f}, cls_loss: {:.5f}, reg_loss: {:.5f}'
                      .format(train_step, loss.item(), avg_losses[-1], loss_cls.item(), loss_reg.item()))
                if visualize is not None:
                    loss_window = visualize.line(X=np.arange(0, train_step + 1, options.report_frequency),
                                           Y=avg_losses,
                                           update='update' if loss_window else None,
                                           win=loss_window,
                                           opts={'title': "Training loss", 'xlabel': "Step", 'ylabel': "Loss"})

            train_step += 1

        epoch_loss = epoch_loss / len(train_loader)
        print('Average epoch loss: {:.5f}'.format(epoch_loss))
        metric = epoch_loss
        if val_loader is not None:
            val_loss, val_accuracy = validate(net, criterion_cls, criterion_reg, val_loader, device, metadata)
            print('Validation loss: {:.5f}, validation accuracy: {:.2f}%'.format(val_loss, val_accuracy))
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            # metric = val_loss
            metric = -val_accuracy

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            # other scheduler types
            scheduler.step()

        # Get current learning rate. Is there any better way?
        lr = None
        for param_group in optimizer.param_groups:
            if param_group['lr'] is not None:
                lr = param_group['lr']
                break
        print('Current learning rate: {}'.format(lr))
        if e % save_every == 0:
            save_dir = options.save_dir or options.model
            save_checkpoint(net, optimizer, save_dir, e, options)


def main():
    print('Start training...')
    print('System info: ', sys.version)
    print('Numpy version: ', np.__version__)
    print('Torch version: ', torch.__version__)
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
    print('Training options: {}'.format(options))
    device = get_device(options.gpu)

    visualize = options.use_visdom
    visualizer = None
    if visualize:
        # 'server' option is needed because of this error: https://github.com/facebookresearch/visdom/issues/490
        visualizer = visdom.Visdom(server=options.visdom_server, env='Train')
        if not visualizer.check_connection:
            print("Visdom server is unreachable. Run `bash server.sh` to start the server.")
            visualizer = None

    metadata = get_input_data(options.metadata)
    out_cls = metadata['num_classes']
    assert out_cls > 0, 'Number of classes has to be > 0'

    hyper_image = torch.load(options.hyper_data_path)
    hyper_labels = torch.load(options.tgt_path)
    norm_inv = torch.load(options.src_norm_multiplier).float()

    hyper_labels_cls = hyper_labels[:, :, :out_cls]
    hyper_labels_reg = hyper_labels[:, :, out_cls:]

    out_reg = hyper_labels_reg.shape[2]

    R, C, num_bands = hyper_image.shape

    train_set, val_set = split_data(R, C, norm_inv, options.patch_size, options.patch_stride)

    # Model construction
    model_name = options.model

    if model_name == 'ChenModel':
        model = ChenModel(num_bands, out_cls, out_reg, patch_size=options.patch_size, n_planes=32)
        loss_cls = nn.BCELoss()
        loss_reg = nn.MSELoss()
    elif model_name == 'PhamModel':
        model = PhamModel(num_bands, out_cls, out_reg, patch_size=options.patch_size, n_planes=32)
        loss_cls = nn.BCELoss()
        loss_reg = nn.MSELoss()
    elif model_name == 'LeeModel':
        model = LeeModel(num_bands, out_cls, out_reg)
        loss_cls = nn.BCELoss()
        loss_reg = nn.MSELoss()

    # loss = nn.BCEWithLogitsLoss()
    # loss = nn.CrossEntropyLoss()
    # loss = nn.MultiLabelSoftMarginLoss(size_average=True)

    train_loader = get_loader(hyper_image,
                              norm_inv,
                              hyper_labels_cls,
                              hyper_labels_reg,
                              train_set,
                              options.batch_size,
                              model_name=model_name,
                              is_3d_convolution=True,
                              patch_size=options.patch_size,
                              shuffle=True)
    val_loader = get_loader(hyper_image,
                            norm_inv,
                            hyper_labels_cls,
                            hyper_labels_reg,
                            val_set,
                            options.batch_size,
                            model_name=model_name,
                            is_3d_convolution=True,
                            patch_size=options.patch_size,
                            shuffle=True)

    # do this before defining the optimizer:  https://pytorch.org/docs/master/optim.html#constructing-it
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=options.lr)

    # End model construction

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        setattr(options, 'start_epoch', checkpoint['epoch'])

    with torch.no_grad():
        print('Model summary: ')
        for input, _, _ in train_loader:
            break

        summary(model,
                input.shape[1:],
                batch_size=options.batch_size,
                device=device.type)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(model)
    print('Classification loss function:', loss_cls)
    print('Regression loss function:', loss_reg)
    print('Scheduler:', scheduler.__dict__)

    # train(model, optimizer, loss_cls, loss_reg, train_loader,
    #       val_loader, device, metadata, options, scheduler=scheduler, visualize=visualizer)
    trainer = Trainer(model, optimizer, loss_cls, loss_reg, scheduler,
                      device, visualizer, metadata, options)
    trainer.train(train_loader, val_loader)
    print('End training...')


if __name__ == "__main__":
    main()
