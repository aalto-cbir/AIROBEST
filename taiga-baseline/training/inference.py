import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import torch.utils.data.dataloader
from input.data_loader import get_loader
from input.focal_loss import FocalLoss
from input.utils import remove_ignored_tasks, get_device, compute_cls_metrics, compute_reg_metrics, compute_data_distribution
from models.model import ChenModel, LeeModel, PhamModel, SharmaModel, HeModel, ModelTrain, PhamModel3layers, \
    PhamModel3layers2, PhamModel3layers3, PhamModel3layers4, PhamModel3layers5, PhamModel3layers6, PhamModel3layers7, \
    PhamModel3layers8, PhamModel3layers9, PhamModel3layers10
from input.data_loader import get_loader
import time


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model_path', type=str,
                        required=True, default='',
                        help="Path to model.")
    parser.add_argument('-save_dir', type=str,
                        required=True, default='',
                        help="Path to save test results.")
    parser.add_argument('-gpu',
                        type=int, default=-1,
                        help="Gpu id to be used, default is -1 which means cpu")

    opt = parser.parse_args()

    return opt


def infer(model, test_loader, device, options, metadata, hyper_labels_reg):
    model.eval()

    pred_cls_logits = torch.tensor([], dtype=torch.float)
    tgt_cls_logits = torch.tensor([], dtype=torch.float)
    all_pred_reg = torch.tensor([], dtype=torch.float)
    all_tgt_reg = torch.tensor([], dtype=torch.float)

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()
    for idx, (src, tgt_cls, tgt_reg, data_idx) in enumerate(test_loader):

        src = src.to(device, dtype=torch.float32)
        tgt_cls = tgt_cls.to(device, dtype=torch.float32)
        tgt_reg = tgt_reg.to(device, dtype=torch.float32)
        categorical = metadata['categorical']

        with torch.no_grad():
            batch_pred_cls, batch_pred_reg = model(src, tgt_cls, tgt_reg, True)

            if not options.no_classification:
                # concat batch predictions
                pred_cls_logits = torch.cat((pred_cls_logits, batch_pred_cls.cpu()), dim=0)
                tgt_cls_logits = torch.cat((tgt_cls_logits, tgt_cls.cpu()), dim=0)

            if not options.no_regression:
                batch_pred_reg = batch_pred_reg.to(torch.device('cpu'))
                tgt_reg = tgt_reg.to(torch.device('cpu'))
                all_tgt_reg = torch.cat((all_tgt_reg, tgt_reg), dim=0)
                all_pred_reg = torch.cat((all_pred_reg, batch_pred_reg), dim=0)

    # balanced_accuracies, avg_accuracy, task_accuracies, conf_matrices = compute_cls_metrics(pred_cls_logits, tgt_cls_logits, options,
    #                                                                        categorical)

    # compute_reg_metrics(test_loader, all_pred_reg, all_tgt_reg, options.epoch, options, metadata,
    #                     hyper_labels_reg, save_dir, should_save=True, mode='test')

    state = {
                'all_target_cls': tgt_cls_logits,
                'all_pred_cls': pred_cls_logits,
                'all_target_reg': all_tgt_reg,
                'all_pred_reg': all_pred_reg
            }
    torch.save(state, '{}/test_pred.pt'.format(save_dir))


def main():
    print('Start testing...')
    start_time = time.time()
    infer_opts = parse_args()

    print('Loading checkpoint from %s' % infer_opts.model_path)
    checkpoint = torch.load(infer_opts.model_path)
    options = checkpoint['options']
    options.epoch = checkpoint['epoch']
    options.save_dir = infer_opts.save_dir
    print('Train opts:', options)

    # options.update(ckpt_options)
    # options.model = ckpt_options.model
    device = get_device(infer_opts.gpu)  # TODO: maybe on CPU?

    metadata = torch.load(options.metadata)
    hyper_image = torch.load(options.hyper_data_path)
    hyper_labels = torch.load(options.tgt_path).float()
    norm_inv = torch.load(options.src_norm_multiplier).float()
    
    elapsed_time = time.time() - start_time
    print('elapsed time: ', elapsed_time)

    # remove ignored tasks
    start_time = time.time()
    hyper_labels_cls, hyper_labels_reg = remove_ignored_tasks(hyper_labels, options, metadata)

    elapsed_time = time.time() - start_time
    print('elapsed time: ', elapsed_time)

    categorical = metadata['categorical']
    print('Metadata values', metadata)
    out_cls = metadata['num_classes']
    out_reg = hyper_labels_reg.shape[-1]
    R, C, num_bands = hyper_image.shape
    test_set = np.load(options.data_split_path + '/test_set.npy', allow_pickle=True)

    elapsed_time = time.time() - start_time
    print('elapsed time: ', elapsed_time)

    print('Data distribution on test set')
#    class_weights = compute_data_distribution(hyper_labels_cls, test_set, categorical)

    # Model construction
    model_name = options.model

    reduction = 'sum' if options.loss_balancing == 'uncertainty' else 'mean'
    loss_reg = nn.MSELoss(reduction=reduction)

    loss_cls_list = []

#    if options.class_balancing == 'cost_sensitive' or options.class_balancing == 'CRL':
#        for i in range(len(categorical.keys())):
#            loss_cls_list.append(nn.CrossEntropyLoss(weight=class_weights[i].to(device)))
#    elif options.class_balancing == 'focal_loss':
#        for i in range(len(categorical.keys())):
#            # loss_cls_list.append(FocalLoss(class_num=len(class_weights[i]), alpha=torch.tensor(class_weights[i]), gamma=2))
#            loss_cls_list.append(FocalLoss(balance_param=class_weights[i].to(device), weight=class_weights[i].to(device)))
#    else:
#        for i in range(len(categorical.keys())):
#            loss_cls_list.append(nn.CrossEntropyLoss())

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

    start_time = time.time()
    test_loader = get_loader(hyper_image,
                             multiplier,
                             hyper_labels_cls,
                             hyper_labels_reg,
                             test_set,
                             32,
                             model_name=model_name,
                             is_3d_convolution=True,
                             augmentation=options.augmentation,
                             patch_size=options.patch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    elapsed_time = time.time() - start_time
    print('elapsed time: ', elapsed_time)                             

    modelTrain = ModelTrain(model, loss_cls_list, loss_reg, metadata, options)
    modelTrain = modelTrain.to(device)
    if checkpoint is not None:
        modelTrain.load_state_dict(checkpoint['model'])
    
    start_time = time.time()

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    infer(modelTrain, test_loader, device, options, metadata, hyper_labels_reg)
    # prof.export_chrome_trace(options.save_dir + '/profiler_log_final.json')
    # print(prof)

    elapsed_time = time.time() - start_time
    print('elapsed time: ', elapsed_time)


if __name__ == "__main__":
    main()
