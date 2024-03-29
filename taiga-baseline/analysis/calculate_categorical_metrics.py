#! /usr/bin/env python3

import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt  
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
            description='Options for running categorical analysis',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred', type=str,
                        required=True, default='',
                        help='Path to predicted values.')

    parser.add_argument('--test_set', type=str,
                        required=True, default='',
                        help='Path to test set.')

    parser.add_argument('--stand_ids', type=str,
                        required=True, default='',
                        help='Path to stand ids.')
    opt = parser.parse_args()

    return opt

options = parse_args()
#pred_test_set = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2_test_set.pt')
pred_test_set = torch.load(options.pred)
# pred_test_set = torch.load('./baseline_model_test_pred.pt')

all_target_cls = pred_test_set['all_target_cls']
all_pred_cls = pred_test_set['all_pred_cls']

#eelis_test_set = np.load('./eelis_test_set_patch_size_13.npy', allow_pickle=True)
test_set = np.load(options.test_set, allow_pickle=True)
#full_stand_id = np.load('../../../inference/stand_ids.npy', allow_pickle=True)
full_stand_id = np.load(options.stand_ids, allow_pickle=True)
stand_id_list = []
for i in range(test_set.shape[0]):
    stand_id_list.append(full_stand_id[test_set[i][0], test_set[i][1]])

# stand_id_list = pred_test_set['stand_id_list']
unique_stand_id_list = np.unique(stand_id_list)
variables = ['fertility_class', 'soil_class', 'main_tree_species_class']
columns = ['standid']
for x in ['', 'pred_']:
    for v in variables:
        columns.append(x + v)

data = []
for i in range(len(unique_stand_id_list)):
    stand_id = unique_stand_id_list[i].astype(np.int32) # get stand_id
    ids = np.where(stand_id_list == unique_stand_id_list[i]) 
    #print("id:", ids)
    # pred_pixel_from_stand = all_pred_cls[ids]
    # target_pixel_from_stand = all_target_cls[ids]
    pred_cls = torch.sum(all_pred_cls[ids], 0)
    #print("all pred cls:", all_pred_cls[ids])
    #print("pred cls:", pred_cls)
    target_cls = torch.sum(all_target_cls[ids], 0)
    #print("all target cls:", all_target_cls[ids])
    #print("target cls:", target_cls)
    row = [stand_id, target_cls[0:4].argmax(-1).item(), target_cls[4:6].argmax(-1).item(),\
           target_cls[6:9].argmax(-1).item(), pred_cls[0:4].argmax(-1).item(),\
           pred_cls[4:6].argmax(-1).item(), pred_cls[6:9].argmax(-1).item()]
    #print(row)
    data.append(row)

df = pd.DataFrame(data, columns = columns)
#print(df)
fertility_class = df['fertility_class'] == df['pred_fertility_class']
soil_class = df['soil_class'] == df['pred_soil_class']
main_tree_species_class = df['main_tree_species_class'] == df['pred_main_tree_species_class']
fertility_class[fertility_class == True]
soil_class[soil_class == True]
main_tree_species_class[main_tree_species_class == True]

accuracy_metrics = ['micro', 'macro']

for var in variables:
    pred_var = 'pred_' + var
    conf_matrix = confusion_matrix(df[var], df[pred_var])
    conf_matrix_old = confusion_matrix(df[var], df[pred_var])
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix = np.around(100 * conf_matrix, decimals=2)
    micro_acc = np.sum(conf_matrix_old.diagonal())/np.sum(conf_matrix_old)*100
    macro_acc = np.around(np.mean(conf_matrix.diagonal()), decimals=2)
    print(var + ' raw confusion matrix: ')
    print(conf_matrix_old)
    print(var + ' normalized confusion matrix: ')
    print(conf_matrix)
    print(var + ' overall ACC (micro): ' + str(micro_acc))
    print(var + ' mean class ACC (macro): ' + str(macro_acc))
    print()

#fertility_class_conf_matrix = confusion_matrix( df['fertility_class'],  df['pred_fertility_class'])
#fertility_class_conf_matrix_old = confusion_matrix( df['fertility_class'],  df['pred_fertility_class'])
#fertility_class_conf_matrix = fertility_class_conf_matrix / fertility_class_conf_matrix.sum(axis=1, keepdims=True)
#fertility_class_conf_matrix = np.around(100 * fertility_class_conf_matrix, decimals=2)
#fertility_class_macro_acc = np.around(np.mean(fertility_class_conf_matrix.diagonal()), decimals=2)
#print(fertility_class_conf_matrix)
#
#
#soil_class_conf_matrix = confusion_matrix( df['soil_class'],  df['pred_soil_class'])
#soil_class_conf_matrix_old = confusion_matrix( df['soil_class'],  df['pred_soil_class'])
#soil_class_conf_matrix = soil_class_conf_matrix / soil_class_conf_matrix.sum(axis=1, keepdims=True)
#soil_class_conf_matrix = np.around(100 * soil_class_conf_matrix, decimals=2)
#soil_class_macro_acc = np.around(np.mean(soil_class_conf_matrix.diagonal()), decimals=2)
#
#
#main_tree_species_class_conf_matrix = confusion_matrix( df['main_tree_species_class'],  df['pred_main_tree_species_class'])
#main_tree_species_class_conf_matrix_old = confusion_matrix( df['main_tree_species_class'],  df['pred_main_tree_species_class'])
#main_tree_species_class_conf_matrix = main_tree_species_class_conf_matrix / main_tree_species_class_conf_matrix.sum(axis=1, keepdims=True)
#main_tree_species_class_conf_matrix = np.around(100 * main_tree_species_class_conf_matrix, decimals=2)
#main_tree_species_class_macro_acc = np.around(np.mean(main_tree_species_class_conf_matrix.diagonal()), decimals=2)
#
#fertility_class_micro_acc = np.sum(fertility_class_conf_matrix_old.diagonal()) / np.sum(fertility_class_conf_matrix_old) * 100
#soil_class_micro_acc = np.sum(soil_class_conf_matrix_old.diagonal()) / np.sum(soil_class_conf_matrix_old) * 100
#main_tree_species_class_micro_acc = np.sum(main_tree_species_class_conf_matrix_old.diagonal()) / np.sum(main_tree_species_class_conf_matrix_old) * 100
#
#print(fertility_class_micro_acc)
#print(soil_class_micro_acc)
#print(main_tree_species_class_micro_acc)
#print((fertility_class_micro_acc + soil_class_micro_acc + main_tree_species_class_micro_acc) / 3)
#
#print('----')
#print(fertility_class_macro_acc)
#print(soil_class_macro_acc)
#print(main_tree_species_class_macro_acc)
#print((fertility_class_macro_acc + soil_class_macro_acc + main_tree_species_class_macro_acc) / 3)






# band descriptions = "fertility_class": {"type": "categorical", "values": {"0": "None", "1": "Herb-rich forest", "2": "Herb-rich heath forest", "3": "Mesic heath forest", "4": "Sub-xeric heath forest", "5": "Xeric heath forest", "6": "Barren heath forest"}, "name": "fertility_class"}, "soil_class": {"type": "categorical", "values": {"0": "None", "1": "Mineral", "2": "Organic"}, "name": "soil_class"}, "main_tree_species_class": {"type": "categorical", "values": {"0": "None", "1": "Scots pine", "2": "Norway spruce", "3": "birch and other broadleaves"}, "name": "main_tree_species_class"}

# Fertility Class
fertility_class_label = ['h-rich', 'mesic', 'sub-x', 'xeric']

cm = confusion_matrix(df['fertility_class'].values,  df['pred_fertility_class'].values)
cm = cm / cm.sum(axis=1, keepdims=True)
cm = np.around(100 * cm, decimals=2)

fig, ax = plt.subplots(figsize=(5, 5))
# ax = fig.add_subplot(111)
cax = ax.matshow(cm, vmin=0, vmax=100)

# fig.colorbar(cax)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#ax.set_xticklabels([''] + fertility_class_label, fontsize=24, rotation=45)
#ax.set_yticklabels([''] + fertility_class_label, fontsize=24)
#plt.tight_layout()
# fig.suptitle('Fertility class at epoch 150')
exit(0)
fig.savefig('conf_matrix_fertility_class_e150.png', format='png')


## Soil Class
soil_class_label = ['miner.', 'org.']
cm = confusion_matrix(df['soil_class'].values,  df['pred_soil_class'].values)
cm = cm / cm.sum(axis=1, keepdims=True)
cm = np.around(100 * cm, decimals=2)

fig, ax = plt.subplots(figsize=(5, 5))
# ax = fig.add_subplot(111)
cax = ax.matshow(cm, vmin=0, vmax=100)

# fig.colorbar(cax)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticklabels([''] + soil_class_label, fontsize=24, rotation=45)
ax.set_yticklabels([''] + soil_class_label, fontsize=24)
# fig.suptitle('Soil class at epoch 150')
plt.tight_layout()
fig.savefig('conf_matrix_soil_class_e150.png', format='png')

## Main Tree Species Class
main_tree_species_class_label = ['pine', 'spruce', 'birch']
cm = confusion_matrix(df['main_tree_species_class'].values,  df['pred_main_tree_species_class'].values)
cm = cm / cm.sum(axis=1, keepdims=True)
cm = np.around(cm, decimals=2)

fig, ax = plt.subplots(figsize=(6, 5))
# ax = fig.add_subplot(111)
cax = ax.matshow(cm, vmin=0, vmax=1)

cbar = fig.colorbar(cax,fraction=0.040, pad=0.04)
cbar.ax.tick_params(labelsize=24)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticklabels([''] + main_tree_species_class_label, fontsize=24, rotation=45)
ax.set_yticklabels([''] + main_tree_species_class_label, fontsize=24)
# fig.suptitle('Main tree species at epoch 150')
plt.tight_layout()
fig.savefig('conf_matrix_main_tree_species_class_e150.png', format='png')
