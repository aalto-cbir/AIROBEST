import numpy as np
import torch
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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

## For Test set S2-45
# test_set = np.load('./eelis_test_set_patch_size_13.npy', allow_pickle=True) # coord of pixel-level test set

# pred_test_set_result = torch.load('./FL_Pham34-271220-gn-aug-eelis-patchsize27_set2.pt') # prediction of pixel-level test set
# all_target_cls = pred_test_set_result['all_target_cls']
# all_pred_cls = pred_test_set_result['all_pred_cls']
# all_target_reg = pred_test_set_result['all_target_reg']
# all_pred_reg = pred_test_set_result['all_pred_reg']

## For complete image

test_set = np.load('../../../inference/complete_image.npy', allow_pickle=True) # coord of pixel-level test set

# prediction 
complete_image1 = torch.load('../../../inference/FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image1.pt')
complete_image2 = torch.load('../../../inference/FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image2.pt')
complete_image3 = torch.load('../../../inference/FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image3.pt')

all_target_cls = torch.cat((complete_image1['all_target_cls'], complete_image2['all_target_cls'], complete_image3['all_target_cls']), 0)
all_pred_cls = torch.cat((complete_image1['all_pred_cls'], complete_image2['all_pred_cls'], complete_image3['all_pred_cls']), 0)
all_target_reg = torch.cat((complete_image1['all_target_reg'], complete_image2['all_target_reg'], complete_image3['all_target_reg']), 0)
all_pred_reg = torch.cat((complete_image1['all_pred_reg'], complete_image2['all_pred_reg'], complete_image3['all_pred_reg']), 0)

full_stand_id = np.load('../../../inference/stand_ids.npy', allow_pickle=True) 

stand_id_list = [] # stand id list of test set
for i in range(test_set.shape[0]):
    stand_id_list.append(full_stand_id[test_set[i][0], test_set[i][1]][0])

unique_stand_id_list = np.unique(stand_id_list)

#### CATEGORICAL EVALUATION ####

categorical_data = []
categorical_columns = ['standid', 'fertility_class', 'soil_class', 'main_tree_species_class', 'pred_fertility_class', 'pred_soil_class', 'pred_main_tree_species_class']
for i in range(len(unique_stand_id_list)):
    print(i)
    stand_id = unique_stand_id_list[i].astype(np.int32) # get stand_id
    ids = np.where(stand_id_list == unique_stand_id_list[i]) 
    pred_cls = torch.sum(all_pred_cls[ids], 0)
    target_cls = torch.sum(all_target_cls[ids], 0)
    row = [stand_id, target_cls[0:4].argmax(-1).item(), target_cls[4:6].argmax(-1).item(), target_cls[6:9].argmax(-1).item(), pred_cls[0:4].argmax(-1).item(), pred_cls[4:6].argmax(-1).item(), pred_cls[6:9].argmax(-1).item()]
    categorical_data.append(row)

categorical_df = pd.DataFrame(categorical_data, columns = categorical_columns)

print('overall accuracy (micro):')
print(overall_accuracy(categorical_df['fertility_class'],  categorical_df['pred_fertility_class']))
print(overall_accuracy(categorical_df['soil_class'],  categorical_df['pred_soil_class']))
print(overall_accuracy(categorical_df['main_tree_species_class'],  categorical_df['pred_main_tree_species_class']))

print('\nmean class accuracy (macro):')
print(mean_class_accuracy(categorical_df['fertility_class'],  categorical_df['pred_fertility_class']))
print(mean_class_accuracy(categorical_df['soil_class'],  categorical_df['pred_soil_class']))
print(mean_class_accuracy(categorical_df['main_tree_species_class'],  categorical_df['pred_main_tree_species_class']))

print('\nprecision_recall_f1 (micro):')
print(micro_precision_recall_f1(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values))
print(micro_precision_recall_f1(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values))
print(micro_precision_recall_f1(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values))

print('\nprecision_recall_f1 (macro):')
print(macro_precision_recall_f1(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values))
print(macro_precision_recall_f1(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values))
print(macro_precision_recall_f1(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values))

print('\nprecision_recall_f1 (weighted):')
print(weighted_precision_recall_f1(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values))
print(weighted_precision_recall_f1(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values))
print(weighted_precision_recall_f1(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values))

print('\nAUC score (macro):')
print(auc_score(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values, 'macro'))
print(auc_score(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values, 'macro'))
print(auc_score(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values, 'macro'))

print('\nAUC score (micro):')
print(auc_score(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values, 'micro'))
print(auc_score(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values, 'micro'))
print(auc_score(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values, 'micro'))

print('\nAUC score (weighted):')
print(auc_score(categorical_df['fertility_class'].values,  categorical_df['pred_fertility_class'].values, 'weighted'))
print(auc_score(categorical_df['soil_class'].values,  categorical_df['pred_soil_class'].values, 'weighted'))
print(auc_score(categorical_df['main_tree_species_class'].values,  categorical_df['pred_main_tree_species_class'].values, 'weighted'))


#### CONTINUOUS EVALUATION ####

all_pred_reg = np.delete(all_pred_reg, [10, 11, 12], 1)
all_target_reg = np.delete(all_target_reg, [10, 11, 12], 1)

columns = ['standid', 'basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index', 'pred_basal_area', 'pred_mean_dbh', 'pred_stem_density', 'pred_mean_height','pred_percentage_of_pine', 'pred_percentage_of_spruce', 'pred_percentage_of_birch', 'pred_woody_biomass', 'pred_leaf_area_index', 'pred_effective_leaf_area_index']
upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45]

continuous_data = []
for i in range(len(unique_stand_id_list)):
    print(i)
    stand_id = unique_stand_id_list[i].astype(np.int32)
    ids = np.where(stand_id_list == unique_stand_id_list[i])
    pred_reg = torch.mean(all_pred_reg[ids], 0) * torch.Tensor(upper_bound)
    target_reg = torch.mean(all_target_reg[ids], 0) * torch.Tensor(upper_bound)
    row = [stand_id] + target_reg.tolist() + pred_reg.tolist()
    continuous_data.append(row)

continuous_df = pd.DataFrame(continuous_data, columns = columns)

print('\nrRMSE:')
print(rrmse(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values))
print(rrmse(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values))
print(rrmse(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values))
print(rrmse(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values))
print(rrmse(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values))
print(rrmse(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values))
print(rrmse(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values))
print(rrmse(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values))
print(rrmse(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values))
print(rrmse(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values))

print('\nRMSE:')
print(rmse(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values))
print(rmse(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values))
print(rmse(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values))
print(rmse(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values))
print(rmse(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values))
print(rmse(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values))
print(rmse(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values))
print(rmse(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values))
print(rmse(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values))
print(rmse(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values))

print('\nrBias:')
print(relative_bias(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values))
print(relative_bias(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values))
print(relative_bias(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values))
print(relative_bias(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values))
print(relative_bias(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values))
print(relative_bias(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values))
print(relative_bias(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values))
print(relative_bias(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values))
print(relative_bias(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values))
print(relative_bias(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values))

print('\nR2 score:')
print(r2_score_func(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values))
print(r2_score_func(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values))
print(r2_score_func(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values))
print(r2_score_func(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values))
print(r2_score_func(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values))
print(r2_score_func(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values))
print(r2_score_func(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values))
print(r2_score_func(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values))
print(r2_score_func(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values))
print(r2_score_func(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values))

print('\nconf_interval 90:')
print(confidence_interval(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values, 0.9))
print(confidence_interval(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values, 0.9))
print(confidence_interval(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values, 0.9))
print(confidence_interval(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values, 0.9))
print(confidence_interval(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values, 0.9))
print(confidence_interval(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values, 0.9))
print(confidence_interval(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values, 0.9))
print(confidence_interval(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values, 0.9))
print(confidence_interval(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values, 0.9))
print(confidence_interval(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values, 0.9))

print('\nconf_interval 95:')
print(confidence_interval(continuous_df['pred_mean_height'].values, continuous_df['mean_height'].values))
print(confidence_interval(continuous_df['pred_basal_area'].values, continuous_df['basal_area'].values))
print(confidence_interval(continuous_df['pred_leaf_area_index'].values, continuous_df['leaf_area_index'].values))
print(confidence_interval(continuous_df['pred_woody_biomass'].values, continuous_df['woody_biomass'].values))
print(confidence_interval(continuous_df['pred_mean_dbh'].values, continuous_df['mean_dbh'].values))
print(confidence_interval(continuous_df['pred_effective_leaf_area_index'].values, continuous_df['effective_leaf_area_index'].values))
print(confidence_interval(continuous_df['pred_stem_density'].values, continuous_df['stem_density'].values))
print(confidence_interval(continuous_df['pred_percentage_of_pine'].values, continuous_df['percentage_of_pine'].values))
print(confidence_interval(continuous_df['pred_percentage_of_spruce'].values, continuous_df['percentage_of_spruce'].values))
print(confidence_interval(continuous_df['pred_percentage_of_birch'].values, continuous_df['percentage_of_birch'].values))