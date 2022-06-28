#! /usr/bin/env python3

import argparse
import numpy as np
import torch
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def parse_args():
    parser = argparse.ArgumentParser(
            description='Options for running the analysis',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred', type=str,
                        required=True, default='',
                        help='Path to predicted values')

    parser.add_argument('--test_set', type=str,
                        required=True, default='',
                        help='Path to test set')

    parser.add_argument('--stand_ids', type=str,
                        required=True, default='',
                        help='Path to stand ids')

    opt = parser.parse_args()

    return opt

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

def confidence_interval_90(pred, target):
    return confidence_interval(pred, target, 0.9)

score_functions = {'rmse': rmse, 'rrmse': rrmse, 'rbias': relative_bias, 'r2': r2_score_func,\
                   'conf_interval_90': confidence_interval_90, 'conf_interval_95': confidence_interval}

options = parse_args()

pred_test_set = torch.load(options.pred)

all_target_reg = pred_test_set['all_target_reg']
all_pred_reg   = pred_test_set['all_pred_reg']

taiga_test_set = np.load(options.test_set, allow_pickle=True)
full_stand_id = np.load(options.stand_ids, allow_pickle=True)
stand_id_list = []
for i in range(taiga_test_set.shape[0]):
    #stand_id_list.append(full_stand_id[eelis_test_set[i][0], eelis_test_set[i][1]][0])
    stand_id_list.append(full_stand_id[taiga_test_set[i][0], taiga_test_set[i][1]])

# stand_id_list = pred_test_set['stand_id_list']
unique_stand_id_list = np.unique(stand_id_list)


all_pred_reg   = np.delete(all_pred_reg,   [10, 11, 12, 13], 1)
all_target_reg = np.delete(all_target_reg, [10, 11, 12, 13], 1)

variables = ['basal_area', 'mean_dbh', 'stem_density', 'mean_height', \
             'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', \
             'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index']
columns = ['standid']

for x in ['', 'pred_']:
    for v in variables:
        columns.append(x + v)

upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45]


hai_data = []
for i in range(len(unique_stand_id_list)):
    stand_id = unique_stand_id_list[i].astype(np.int32)
    ids = np.where(stand_id_list == unique_stand_id_list[i])
    pred_reg = torch.mean(all_pred_reg[ids], 0) * torch.Tensor(upper_bound)
    target_reg = torch.mean(all_target_reg[ids], 0) * torch.Tensor(upper_bound)
    row = [stand_id] + target_reg.tolist() + pred_reg.tolist()
    hai_data.append(row)

df = pd.DataFrame(hai_data, columns = columns)

# d['unique_stand_id_list'] = torch.Tensor(unique_stand_id_list.tolist())
# d['pred_test_set'] = torch.Tensor(c)
# torch.save(d, 'test.pt')

# eelis_standid = data[data.set == 'holdout']['standid'].values
# hai_standid = unique_stand_id_list
# intersection_standid = list(set(eelis_standid).intersection(hai_standid))

# for i in range(len(intersection_standid)):
#     stand_id = intersection_standid[i]

for s in score_functions.keys():
    for v in variables:
        print(s, v, score_functions[s](df['pred_' + v].values, df[v].values))
    print()
