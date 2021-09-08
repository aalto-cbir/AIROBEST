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
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from os import listdir
from os.path import isfile, join
from pathlib import Path
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def rmse(pred, target):
    return sqrt(mean_squared_error(target, pred))

def rrmse(pred, target):
    return sqrt(mean_squared_error(target, pred)) / np.mean(target)

def relative_bias(pred, target):
    return sum(pred - target) / sum(target) 

def r2_score_func(pred, target):
    return r2_score(target, pred)

def confidence_interval(pred, target, confidence= 0.95):
    signed_error = pred - target
    # a = 1.0 * np.array(signed_error)
    n = len(signed_error)
    sorted = np.sort(signed_error)
    return sorted[(int)(n * (1 + confidence)) // 2] - sorted[(int)(n * (1 - confidence)) // 2]

def mean_class_accuracy(target, pred): # macro
    conf_matrix = confusion_matrix(target,  pred)
    # print(conf_matrix)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix = np.around(100 * conf_matrix, decimals=2)
    macro_acc = np.around(np.mean(conf_matrix.diagonal()), decimals=2)
    return macro_acc

def overall_accuracy(target, pred): # micro
    conf_matrix = confusion_matrix(target,  pred)
    micro_acc = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix) * 100
    return micro_acc

def macro_precision_recall_f1(target, pred):
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return precision, recall, f1

def micro_precision_recall_f1(target, pred):
    precision = precision_score(target, pred, average='micro')
    recall = recall_score(target, pred, average='micro')
    f1 = f1_score(target, pred, average='micro')
    return precision, recall, f1

def weighted_precision_recall_f1(target, pred):
    precision = precision_score(target, pred, average='weighted')
    recall = recall_score(target, pred, average='weighted')
    f1 = f1_score(target, pred, average='weighted')
    return precision, recall, f1

def auc_score(target, pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(target)
    target = lb.transform(target)
    pred = lb.transform(pred)
    return roc_auc_score(target, pred, average=average)


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


def infer(model, loader, device, options, metadata, hyper_labels_reg, set_name, stand_id_list, unique_stand_id_list):
    model.eval()

    all_pred_cls = torch.tensor([], dtype=torch.float)
    all_target_cls = torch.tensor([], dtype=torch.float)
    all_pred_reg = torch.tensor([], dtype=torch.float)
    all_target_reg = torch.tensor([], dtype=torch.float)

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()
    for idx, (src, tgt_cls, tgt_reg, data_idx) in enumerate(loader):

        src = src.to(device, dtype=torch.float32)
        tgt_cls = tgt_cls.to(device, dtype=torch.float32)
        tgt_reg = tgt_reg.to(device, dtype=torch.float32)
        categorical = metadata['categorical']

        with torch.no_grad():
            batch_pred_cls, batch_pred_reg = model(src, tgt_cls, tgt_reg, True)

            if not options.no_classification:
                # concat batch predictions
                all_pred_cls = torch.cat((all_pred_cls, batch_pred_cls.cpu()), dim=0)
                all_target_cls = torch.cat((all_target_cls, tgt_cls.cpu()), dim=0)

            if not options.no_regression:
                batch_pred_reg = batch_pred_reg.to(torch.device('cpu'))
                tgt_reg = tgt_reg.to(torch.device('cpu'))
                all_target_reg = torch.cat((all_target_reg, tgt_reg), dim=0)
                all_pred_reg = torch.cat((all_pred_reg, batch_pred_reg), dim=0)

    # balanced_accuracies, avg_accuracy, task_accuracies, conf_matrices = compute_cls_metrics(all_pred_cls, all_target_cls, options,
    #                                                                        categorical)

    # compute_reg_metrics(loader, all_pred_reg, all_target_reg, options.epoch, options, metadata,
    #                     hyper_labels_reg, save_dir, should_save=True, mode='test')

    state = {
                'all_target_cls': all_target_cls,
                'all_pred_cls': all_pred_cls,
                'all_target_reg': all_target_reg,
                'all_pred_reg': all_pred_reg
            }
    # torch.save(state, '{}/train_pred.pt'.format(save_dir))

    
    

    all_pred_reg = np.delete(all_pred_reg, [10, 11, 12], 1)
    all_target_reg = np.delete(all_target_reg, [10, 11, 12], 1)

    continuous_columns = ['standid', 'basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index', 'pred_basal_area', 'pred_mean_dbh', 'pred_stem_density', 'pred_mean_height','pred_percentage_of_pine', 'pred_percentage_of_spruce', 'pred_percentage_of_birch', 'pred_woody_biomass', 'pred_leaf_area_index', 'pred_effective_leaf_area_index']
    upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45]
    continuous_data = []

    categorical_columns = ['standid', 'fertility_class', 'soil_class', 'main_tree_species_class', 'pred_fertility_class', 'pred_soil_class', 'pred_main_tree_species_class']
    categorical_data = []
    
    for i in range(len(unique_stand_id_list)):
        stand_id = unique_stand_id_list[i].astype(np.int32)
        ids = np.where(stand_id_list == unique_stand_id_list[i])

        pred_reg = torch.mean(all_pred_reg[ids], 0) * torch.Tensor(upper_bound)
        target_reg = torch.mean(all_target_reg[ids], 0) * torch.Tensor(upper_bound)
        row = [stand_id] + target_reg.tolist() + pred_reg.tolist()
        continuous_data.append(row)

        pred_cls = torch.sum(all_pred_cls[ids], 0)
        target_cls = torch.sum(all_target_cls[ids], 0)
        row = [stand_id, target_cls[0:4].argmax(-1).item(), target_cls[4:6].argmax(-1).item(), target_cls[6:9].argmax(-1).item(), pred_cls[0:4].argmax(-1).item(), pred_cls[4:6].argmax(-1).item(), pred_cls[6:9].argmax(-1).item()]
        categorical_data.append(row)

    print(set_name)
    continuous_df = pd.DataFrame(continuous_data, columns = continuous_columns)
    rrmseList = [rrmse(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values), rrmse(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values), rrmse(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values), rrmse(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values)]
    print('average rrmse:', np.mean(rrmseList))

    categorical_df = pd.DataFrame(categorical_data, columns = categorical_columns)
    overall_accuracy_list = [overall_accuracy(categorical_df['fertility_class'],  categorical_df['pred_fertility_class']), overall_accuracy(categorical_df['soil_class'],  categorical_df['pred_soil_class']), overall_accuracy(categorical_df['main_tree_species_class'],  categorical_df['pred_main_tree_species_class'])]
    print('average overall accuracy (micro)', np.mean(overall_accuracy_list))

    mean_class_accuracy_list = [mean_class_accuracy(categorical_df['fertility_class'],  categorical_df['pred_fertility_class']), mean_class_accuracy(categorical_df['soil_class'],  categorical_df['pred_soil_class']), mean_class_accuracy(categorical_df['main_tree_species_class'],  categorical_df['pred_main_tree_species_class'])]
    print('average mean class accuracy (macro)', np.mean(mean_class_accuracy_list))


def main():
    print('Start testing...')
    start_time = time.time()
    infer_opts = parse_args()

    print('Loading checkpoint from %s' % infer_opts.model_path)
    
        
    checkpoint = torch.load(infer_opts.model_path)
    options = checkpoint['options']
    # options.epoch = checkpoint['epoch']
    # options.save_dir = infer_opts.save_dir
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
    train_set = np.load(options.data_split_path + '/train_set.npy', allow_pickle=True)
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
    train_loader = get_loader(hyper_image,
                             multiplier,
                             hyper_labels_cls,
                             hyper_labels_reg,
                             train_set,
                             32,
                             model_name=model_name,
                             is_3d_convolution=True,
                             augmentation=options.augmentation,
                             patch_size=options.patch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
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

    eelis_test_set = np.load('./data/TAIGA/eelis-split13-2/test_set.npy', allow_pickle=True)
    eelis_train_set = np.load('./data/TAIGA/eelis-split13-2/train_set.npy', allow_pickle=True)
    full_stand_id = np.load('./stand_ids/stand_ids.npy', allow_pickle=True)

    test_stand_id_list = []
    for i in range(eelis_test_set.shape[0]):
        test_stand_id_list.append(full_stand_id[eelis_test_set[i][0], eelis_test_set[i][1]][0])
    unique_test_stand_id_list = np.unique(test_stand_id_list)

    train_stand_id_list = []
    for i in range(eelis_train_set.shape[0]):
        train_stand_id_list.append(full_stand_id[eelis_train_set[i][0], eelis_train_set[i][1]][0])
    unique_train_stand_id_list = np.unique(train_stand_id_list)


    modelTrain = ModelTrain(model, loss_cls_list, loss_reg, metadata, options)
    modelTrain = modelTrain.to(device)
    # files = [f for f in listdir('./checkpoint/baseline_model/') if isfile(join('./checkpoint/baseline_model/', f))]
    paths = sorted(Path('./checkpoint/baseline_model/').iterdir(), key=os.path.getmtime)
    for f in paths:
        checkpoint = torch.load(f)
        print('Loading checkpoint:', f)

        # if checkpoint is not None:
        modelTrain.load_state_dict(checkpoint['model'])

        infer(modelTrain, train_loader, device, options, metadata, hyper_labels_reg, 'train', train_stand_id_list, unique_train_stand_id_list)
        infer(modelTrain, test_loader, device, options, metadata, hyper_labels_reg, 'test', test_stand_id_list, unique_test_stand_id_list)
        

if __name__ == "__main__":
    main()
