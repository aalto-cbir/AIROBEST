import numpy as np
import torch
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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

# pred_test_set = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2_train_pred.pt')
pred_test_set = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2_test_set.pt')

all_target_reg = pred_test_set['all_target_reg']
all_pred_reg = pred_test_set['all_pred_reg']

eelis_test_set = np.load('./eelis_test_set_patch_size_13.npy', allow_pickle=True)
full_stand_id = np.load('../../../inference/stand_ids.npy', allow_pickle=True)
stand_id_list = []
for i in range(eelis_test_set.shape[0]):
    stand_id_list.append(full_stand_id[eelis_test_set[i][0], eelis_test_set[i][1]][0])

# stand_id_list = pred_test_set['stand_id_list']
unique_stand_id_list = np.unique(stand_id_list)



all_pred_reg = np.delete(all_pred_reg, [10, 11, 12], 1)
all_target_reg = np.delete(all_target_reg, [10, 11, 12], 1)

columns = ['standid', 'basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index', 'pred_basal_area', 'pred_mean_dbh', 'pred_stem_density', 'pred_mean_height','pred_percentage_of_pine', 'pred_percentage_of_spruce', 'pred_percentage_of_birch', 'pred_woody_biomass', 'pred_leaf_area_index', 'pred_effective_leaf_area_index']
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

print('Stand-wise:')

print(str(rrmse(df['pred_mean_height'].values, df['mean_height'].values)).replace('.', '.'))
print(str(rrmse(df['pred_basal_area'].values, df['basal_area'].values)).replace('.', '.'))
print(str(rrmse(df['pred_leaf_area_index'].values, df['leaf_area_index'].values)).replace('.', '.'))
print(str(rrmse(df['pred_woody_biomass'].values, df['woody_biomass'].values)).replace('.', '.'))
print(str(rrmse(df['pred_mean_dbh'].values, df['mean_dbh'].values)).replace('.', '.'))
print(str(rrmse(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values)).replace('.', '.'))
print(str(rrmse(df['pred_stem_density'].values, df['stem_density'].values)).replace('.', '.'))
print(str(rrmse(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values)).replace('.', '.'))
print(str(rrmse(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values)).replace('.', '.'))
print(str(rrmse(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)).replace('.', '.'))
print('\n')

print(str(rmse(df['pred_mean_height'].values, df['mean_height'].values)).replace('.', '.'))
print(str(rmse(df['pred_basal_area'].values, df['basal_area'].values)).replace('.', '.'))
print(str(rmse(df['pred_leaf_area_index'].values, df['leaf_area_index'].values)).replace('.', '.'))
print(str(rmse(df['pred_woody_biomass'].values, df['woody_biomass'].values)).replace('.', '.'))
print(str(rmse(df['pred_mean_dbh'].values, df['mean_dbh'].values)).replace('.', '.'))
print(str(rmse(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values)).replace('.', '.'))
print(str(rmse(df['pred_stem_density'].values, df['stem_density'].values)).replace('.', '.'))
print(str(rmse(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values)).replace('.', '.'))
print(str(rmse(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values)).replace('.', '.'))
print(str(rmse(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)).replace('.', '.'))
print('\n')

print(str(relative_bias(df['pred_mean_height'].values, df['mean_height'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_basal_area'].values, df['basal_area'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_leaf_area_index'].values, df['leaf_area_index'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_woody_biomass'].values, df['woody_biomass'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_mean_dbh'].values, df['mean_dbh'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_stem_density'].values, df['stem_density'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values)).replace('.', '.'))
print(str(relative_bias(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)).replace('.', '.'))
print('\n')

print(str(r2_score_func(df['pred_mean_height'].values, df['mean_height'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_basal_area'].values, df['basal_area'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_leaf_area_index'].values, df['leaf_area_index'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_woody_biomass'].values, df['woody_biomass'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_mean_dbh'].values, df['mean_dbh'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_stem_density'].values, df['stem_density'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values)).replace('.', '.'))
print(str(r2_score_func(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)).replace('.', '.'))
print('\n')

print(str(confidence_interval(df['pred_mean_height'].values, df['mean_height'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_basal_area'].values, df['basal_area'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_leaf_area_index'].values, df['leaf_area_index'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_woody_biomass'].values, df['woody_biomass'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_mean_dbh'].values, df['mean_dbh'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_stem_density'].values, df['stem_density'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values, 0.9)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values, 0.9)).replace('.', '.'))
print('\n')

print(str(confidence_interval(df['pred_mean_height'].values, df['mean_height'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_basal_area'].values, df['basal_area'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_leaf_area_index'].values, df['leaf_area_index'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_woody_biomass'].values, df['woody_biomass'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_mean_dbh'].values, df['mean_dbh'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_stem_density'].values, df['stem_density'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values)).replace('.', '.'))
print(str(confidence_interval(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)).replace('.', '.'))

print('\n')
rrmseList = [rrmse(df['pred_mean_height'].values, df['mean_height'].values), rrmse(df['pred_basal_area'].values, df['basal_area'].values), rrmse(df['pred_leaf_area_index'].values, df['leaf_area_index'].values), rrmse(df['pred_woody_biomass'].values, df['woody_biomass'].values), rrmse(df['pred_mean_dbh'].values, df['mean_dbh'].values), rrmse(df['pred_effective_leaf_area_index'].values, df['effective_leaf_area_index'].values), rrmse(df['pred_stem_density'].values, df['stem_density'].values), rrmse(df['pred_percentage_of_pine'].values, df['percentage_of_pine'].values), rrmse(df['pred_percentage_of_spruce'].values, df['percentage_of_spruce'].values), rrmse(df['pred_percentage_of_birch'].values, df['percentage_of_birch'].values)]
print(np.mean(rrmseList))

# ###########################################

# columns = ['basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index', 'pred_basal_area', 'pred_mean_dbh', 'pred_stem_density', 'pred_mean_height','pred_percentage_of_pine', 'pred_percentage_of_spruce', 'pred_percentage_of_birch', 'pred_woody_biomass', 'pred_leaf_area_index', 'pred_effective_leaf_area_index']
# upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45]
# pixel_wise_data = []
# for i in range(len(all_target_reg)):
#     row = (all_target_reg[i].numpy() * np.array(upper_bound)).tolist() + (all_pred_reg[i].numpy() * np.array(upper_bound)).tolist()
#     pixel_wise_data.append(row)

# pixel_wise_df = pd.DataFrame(pixel_wise_data, columns = columns)

# print('\nPixel-wise:')
# # 0basal_area, 1mean_dbh, 2stem_density, 3mean_height, 4woody_biomass, 5leaf_area_index, 6effective_leaf_area_index

# print(str(rrmse(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values)).replace('.', '.'))
# print(str(rrmse(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values)).replace('.', '.'))
# print('\n')

# print(str(rmse(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values)).replace('.', '.'))
# print(str(rmse(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values)).replace('.', '.'))
# print('\n')

# print(str(relative_bias(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values)).replace('.', '.'))
# print(str(relative_bias(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values)).replace('.', '.'))
# print('\n')

# print(str(r2_score_func(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values)).replace('.', '.'))
# print(str(r2_score_func(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values)).replace('.', '.'))
# print('\n')

# print(str(confidence_interval(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values, 0.9)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values, 0.9)).replace('.', '.'))
# print('\n')

# print(str(confidence_interval(pixel_wise_df['pred_mean_height'].values, pixel_wise_df['mean_height'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_basal_area'].values, pixel_wise_df['basal_area'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_leaf_area_index'].values, pixel_wise_df['leaf_area_index'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_woody_biomass'].values, pixel_wise_df['woody_biomass'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_mean_dbh'].values, pixel_wise_df['mean_dbh'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_effective_leaf_area_index'].values, pixel_wise_df['effective_leaf_area_index'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_stem_density'].values, pixel_wise_df['stem_density'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_pine'].values, pixel_wise_df['percentage_of_pine'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_spruce'].values, pixel_wise_df['percentage_of_spruce'].values)).replace('.', '.'))
# print(str(confidence_interval(pixel_wise_df['pred_percentage_of_birch'].values, pixel_wise_df['percentage_of_birch'].values)).replace('.', '.'))