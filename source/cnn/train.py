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

from models.model import ChenModel, LeeModel
from input.utils import split_data
from input.data_loader import get_loader


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_path',
                        required=False, type=str,
                        default='../../data/hyperspectral_src_l2norm.pt',
                        help='Path to training input file')
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
    train.add_argument('-patch_step', type=int,
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


def save_checkpoint(model, optimizer, model_name, epoch, shard_id, options):
    """
    Saving model's state dict
    TODO: also save optimizer' state dict and model options and enable restoring model from last training step
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
        'shard_id': shard_id,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'options': options
    }
    torch.save(state, '{}/{}_e{}_{}.pt'.format(path, model_name, epoch, shard_id))
    print('Saving model at epoch %d shard %d' % (epoch, shard_id))


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
            loss = loss_cls + 3 * loss_reg

            sum_loss += loss.item()
            sum_accuracy += compute_accuracy(pred_cls, tgt_cls, metadata)

    # return average validation loss
    average_loss = sum_loss / len(val_loader)
    accuracy = sum_accuracy * 100 / len(val_loader)
    return average_loss, accuracy




def build_loader(metadata, options, shard_id):
    out_cls = metadata['num_classes']
    assert out_cls > 0, 'Number of classes has to be > 0'

    hyper_image = torch.load("%s_%d.pt" % (options.src_path, shard_id))
    hyper_labels = torch.load("%s_%d.pt" % (options.tgt_path, shard_id))
    hyper_labels_cls = hyper_labels[:, :, :out_cls]
    hyper_labels_reg = hyper_labels[:, :, out_cls:]

    R, C, B = hyper_image.shape
    train_set, test_set, val_set = split_data(R, C, options.patch_size, shard_id, options.patch_step)

    train_loader = get_loader(hyper_image,
                              hyper_labels_cls,
                              hyper_labels_reg,
                              train_set,
                              options.batch_size,
                              model_name=options.model,
                              is_3d_convolution=True,
                              patch_size=options.patch_size,
                              shuffle=True)
    val_loader = get_loader(hyper_image,
                            hyper_labels_cls,
                            hyper_labels_reg,
                            val_set,
                            options.batch_size,
                            model_name=options.model,
                            is_3d_convolution=True,
                            patch_size=options.patch_size,
                            shuffle=True)
    return train_loader, val_loader


def train(net, optimizer, criterion_cls, criterion_reg, device, metadata,
          options, scheduler=None, visualize=None):
    """
    Training

    TODO: checkpoint
    :param net:
    :param optimizer:
    :param criterion_cls:
    :param criterion_reg:
    :param device:
    :param metadata:
    :param options:
    :param scheduler:
    :param visualize:
    :return:
    """
    epoch = options.epoch
    start_epoch = options.start_epoch + 1 if 'start_epoch' in options else 1
    save_every = 1  # specify number of shards to save model
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

        for id in range(metadata['num_shards']):
            shard_loss = 0.0
            train_loader, val_loader = build_loader(metadata, options, id)

            if e == 0 and id == 0:
                with torch.no_grad():
                    print('Model summary: ')
                    for input, _, _ in train_loader:
                        break

                    summary(net,
                            input.shape[1:],
                            batch_size=options.batch_size,
                            device=device.type)

            for idx, (src, tgt_cls, tgt_reg) in enumerate(train_loader):
                src = src.to(device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(device, dtype=torch.float32)
                # tgt = tgt.to(device, dtype=torch.int64)

                optimizer.zero_grad()
                pred_cls, pred_reg = net(src)
                loss_cls = criterion_cls(pred_cls, tgt_cls)
                loss_reg = criterion_reg(pred_reg, tgt_reg)
                loss = loss_cls + 3 * loss_reg

                sum_loss += loss.item()
                shard_loss += loss.item()
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if train_step % options.report_frequency == 0:
                    # TODO: with LeeModel, take average of the loss
                    avg_losses.append(np.mean(losses[-100:]))
                    print('Training loss at step {}: {:.8f}, average loss: {:.8f}, cls_loss: {:.8f}, reg_loss: {:.8f}'
                          .format(train_step, loss.item(), avg_losses[-1], loss_cls.item(), loss_reg.item()))
                    if visualize is not None:
                        loss_window = visualize.line(X=np.arange(0, train_step + 1, options.report_frequency),
                                                     Y=avg_losses,
                                                     update='update' if loss_window else None,
                                                     win=loss_window,
                                                     opts={'title': "Training loss", 'xlabel': "Step",
                                                           'ylabel': "Loss"})

                train_step += 1

            shard_loss = shard_loss / len(train_loader)
            print('Average loss of shard {}: {:.5f}'.format(id, shard_loss))
            metric = shard_loss
            if val_loader is not None:
                val_loss, val_accuracy = validate(net, criterion_cls, criterion_reg, val_loader, device, metadata)
                print('Validation loss: {:.8f}, validation accuracy: {:.8f}%'.format(val_loss, val_accuracy))
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                metric = val_loss
                # metric = -val_accuracy

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
            if id % save_every == 0:
                save_dir = options.save_dir or options.model
                save_checkpoint(net, optimizer, save_dir, e, id, options)

        # TODO: epoch validation


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
        print('Overwrite options with values from checkpoint!!!')
        checkpoint = torch.load(options.train_from)
        options = checkpoint['options']

    # TODO: check for minimum patch_size
    print('Training options: {}'.format(options))
    device = get_device(options.gpu)

    visualize = options.use_visdom
    if visualize:
        # 'server' option is needed because of this error: https://github.com/facebookresearch/visdom/issues/490
        vis = visdom.Visdom(server=options.visdom_server, env='Train')
        if not vis.check_connection:
            print("Visdom server is unreachable. Run `bash server.sh` to start the server.")
            vis = None

    metadata = get_input_data(options.metadata)
    out_cls = metadata['num_classes']
    out_reg = metadata['num_regressions']
    assert out_cls > 0, 'Number of classes has to be > 0'

    W, H, num_bands = metadata['image_shape']

    model_name = options.model

    if model_name == 'ChenModel':
        model = ChenModel(num_bands, out_cls, out_reg, patch_size=options.patch_size, n_planes=32)
        loss_cls = nn.BCELoss()
        loss_reg = nn.MSELoss()
    elif model_name == 'LeeModel':
        model = LeeModel(num_bands, out_cls)
        loss_cls = nn.BCELoss()
        loss_reg = nn.MSELoss()

    # loss = nn.MultiLabelSoftMarginLoss(size_average=True)

    # do this before defining the optimizer:  https://pytorch.org/docs/master/optim.html#constructing-it
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=options.lr)

    # End model construction

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        setattr(options, 'start_epoch', checkpoint['epoch'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print(model)
    print('Classification loss function:', loss_cls)
    print('Regression loss function:', loss_reg)
    print('Scheduler:', scheduler.__dict__)

    train(model, optimizer, loss_cls, loss_reg,
          device, metadata, options, scheduler=scheduler, visualize=vis)
    print('End training...')


if __name__ == "__main__":
    main()
