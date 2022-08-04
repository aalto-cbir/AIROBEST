#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import scipy.stats as stats
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
import random
from sklearn.model_selection import train_test_split
import copy

data_path = '../../data/TAIGA/'
save_path = data_path + 'data-split-aug-horizontal/'

splitting_patch_size = 13

flip = True
flip_mode = "horizontal" # can be horizontal, vertical, or mixed

random_state = 797  # 646, 919, 390, 101
print('Random seed:', random_state)

def get_patch(tensor, row, col, patch_size):
    row1, col1 = row - patch_size // 2, col - patch_size // 2
    row2, col2 = row + patch_size // 2, col + patch_size // 2
    return tensor[row1:(row2 + 1), col1:(col2 + 1)]

def split_data(rows, cols, mask, hyper_labels_cls, hyper_labels_reg, patch_size, stride=1, mode='grid', is_mask = False):
    train = []
    val = []
    test = []
    origin = []
    coords = []
    R, C = mask.shape
    if mode == 'grid':
        stride = patch_size
    for i in range(patch_size // 2, rows - patch_size // 2, stride):
        for j in range(patch_size // 2, cols - patch_size // 2, stride):
            patch = get_patch(mask, i, j, patch_size)
            if torch.min(patch) > 0:  # make sure there is no black pixels in the patch
                if mode == 'random' or mode == 'grid':
                    coords.append([i, j, 0, None, None])
    if mode == 'random' or mode == 'grid':
        train, val = train_test_split(coords, train_size=0.9, random_state=random_state, shuffle=True)

    if flip: 
        print("Adding flipped samples")
        new_data = []
        for sample in train:
            r, c, _, _, _ = sample
            if flip_mode == "horizontal":
                code = 1
            elif flip_mode == "vertical":
                code = 2
            else:
                code = 1 if np.random.random() > 0.5 else 2
            new_data.append([r, c, code, None, None])
        
        train = train + new_data

    np.random.seed(123)
    np.random.shuffle(train)
    train = np.array(train)
    val = np.array(val)
    coords = np.array(coords)
    #print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    # print('Train', train[0:10])
    # print('Val', val[0:10])
    return train, test, val, coords

def draw2(arr, title, save_path = "./out/", cmap="viridis", isCustomCmap = False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if isCustomCmap is True:
        # clist = [(0,"gray"), (1./100.,"blue"), (495./1000., "white"),  (1, "red")]
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
        oldcmap = matplotlib.cm.get_cmap(cmap, 256)
        newcolors = oldcmap(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        newcolors[:1, :] = black
        cmap = ListedColormap(newcolors)
    plt.imsave(save_path + title, arr, cmap=cmap)


forest_stand_data = pd.read_csv('./taiga_forest_stand_data.csv')

eelis_standid = forest_stand_data['standid'].tolist()
eelis_set = forest_stand_data['set'].tolist()

complete_stand_id = np.load('../../data/TAIGA/stand_ids.npy').squeeze()
#bad_stand_list = pd.read_csv('./bad_stands_updated_15092020.csv')['standid'].tolist()
#for bad_stand_id in bad_stand_list:
#    complete_stand_id[complete_stand_id == bad_stand_id] = 0
# unique_stand_ids = np.unique(complete_stand_id)

mask = np.load('../../data/TAIGA/mask.npy').squeeze()

# # Hai mask
#print("stand id shape: ", complete_stand_id.shape)
#print("mask shape: ", mask.shape)
hai_mask = np.divide(complete_stand_id * mask, mask, out=np.zeros_like(complete_stand_id * mask), where=mask!=0)
draw2(hai_mask, "hai_mask.png", save_path, "rainbow", True)

R, C = hai_mask.shape
#print("hai mask shape:", hai_mask.shape)
np.save(save_path + 'hai_mask.npy', hai_mask)

# Eelis mask
eelis_mask = copy.deepcopy(hai_mask)

for i in range(len(eelis_standid)):
    if eelis_set[i] == 'training':
        eelis_mask[eelis_mask == eelis_standid[i]] = -1 
    else:
        eelis_mask[eelis_mask == eelis_standid[i]] = -2 

eelis_mask[eelis_mask > -1] = 0
eelis_mask[eelis_mask == -1] = 1 # training 1
eelis_mask[eelis_mask == -2] = 2 # holdout / test 2

draw2(eelis_mask, "eelis_mask.png", save_path, "rainbow", True)


# for i in range(len(eelis_standid)):
#     print(eelis_standid[i], eelis_set[i], np.where(hai_stand_id.flatten() == eelis_standid[i])[0].shape[0] > 0)

# option 1: based on Eelis's stand

eelis_train_stand_mask = copy.deepcopy(eelis_mask)
eelis_train_stand_mask[eelis_train_stand_mask == 2] = 0

draw2(eelis_train_stand_mask, "eelis_train_stand_mask.png", save_path, "rainbow", True)

eelis_test_stand_mask = copy.deepcopy(eelis_mask)
eelis_test_stand_mask[eelis_test_stand_mask == 1] = 0

draw2(eelis_test_stand_mask, "eelis_test_stand_mask.png", save_path, "rainbow", True)

np.save(save_path + 'eelis_train_stand_mask.npy', eelis_train_stand_mask)
#print("eelis train stand mask shape", eelis_train_stand_mask.shape)

np.save(save_path + 'eelis_test_stand_mask.npy', eelis_test_stand_mask)
#print("eelis test stand mask shape", eelis_test_stand_mask.shape)

eelis_train_set, _, eelis_val_set, eelis_origin_set = split_data(R, C, torch.Tensor(eelis_train_stand_mask), [], [], splitting_patch_size, splitting_patch_size, 'grid', False)

_, _, _, eelis_test_set = split_data(R, C, torch.Tensor(eelis_test_stand_mask), [], [], splitting_patch_size, splitting_patch_size, 'grid', False)


np.save(save_path + '/train_set.npy', eelis_train_set)
np.save(save_path + '/test_set.npy', eelis_test_set)
np.save(save_path + '/val_set.npy', eelis_val_set)

#stands_origin = set()
#for x, y, _, _, _ in eelis_origin_set:
#    stands_origin.add(complete_stand_id[x, y])
#print("origin stands:", len(stands_origin))

stands_train = set()
stands_train_val = set()
for x, y, _, _, _ in eelis_train_set:
    stands_train.add(complete_stand_id[x, y])
    stands_train_val.add(complete_stand_id[x, y])
#print("train stands:", len(stands_train))

stands_val = set()
for x, y, _, _, _ in eelis_val_set:
    stands_val.add(complete_stand_id[x, y])
    stands_train_val.add(complete_stand_id[x, y])
#print("val stands:", len(stands_val))
print("train & val set stands:", len(stands_train_val))

stands_test = set()
for x, y, _, _, _ in eelis_test_set:
    stands_test.add(complete_stand_id[x, y])
print("test set stands:", len(stands_test))

#print("origin set:", eelis_origin_set.shape)
print("train set patches:", eelis_train_set.shape[0])
print("val set patches:", eelis_val_set.shape[0])
print("test set patches:", eelis_test_set.shape[0])
print()
print("Files saved to", save_path)
print("End splitting.")

# ----- 
# eelis_train_set = np.load('./eelis_train_set_patch_size_13.npy', allow_pickle=True)
# eelis_test_set = np.load('./eelis_test_set_patch_size_13.npy', allow_pickle=True)
# eelis_val_set = np.load('./eelis_val_set_patch_size_13.npy', allow_pickle=True)

# columns = ['Y', 'X', 'set']
# Y = eelis_train_set[:,0].tolist() + eelis_val_set[:, 0].tolist() + eelis_test_set[:, 0].tolist() 
# X = eelis_train_set[:,1].tolist() + eelis_val_set[:, 1].tolist() + eelis_test_set[:, 1].tolist()
# set_label = ['train']*len(eelis_train_set[:,0].tolist()) + ['val']*len(eelis_val_set[:,0].tolist()) + ['test']*len(eelis_test_set[:,0].tolist())

# df = pd.DataFrame([Y, X, set_label]).T
# df.columns = columns
# df.to_csv('split_of_eelis_stand_distribution.csv', index=False)

# train_patch_mask = np.zeros((12143, 12826))
# patch_size = 45

# for i in range(df.shape[0]):
#     row = df['Y'][i]
#     col = df['X'][i]
#     row1, col1 = row - patch_size // 2, col - patch_size // 2
#     row2, col2 = row + patch_size // 2, col + patch_size // 2
#     if (df['set'][i] == 'train'):
#         train_patch_mask[row1:(row2 + 1), col1:(col2 + 1)] += 1
#     elif (df['set'][i] == 'val'):
#         train_patch_mask[row1:(row2 + 1), col1:(col2 + 1)] += 1



# test_patch_mask = np.zeros((12143, 12826))
# for i in range(df.shape[0]):
#     row = df['Y'][i]
#     col = df['X'][i]
#     row1, col1 = row - patch_size // 2, col - patch_size // 2
#     row2, col2 = row + patch_size // 2, col + patch_size // 2
#     if (df['set'][i] == 'test'):
#         test_patch_mask[row1:(row2 + 1), col1:(col2 + 1)] -= 1

# for i in range(df.shape[0]):
#     row = df['Y'][i]
#     col = df['X'][i]
#     row1, col1 = row - patch_size // 2, col - patch_size // 2
#     row2, col2 = row + patch_size // 2, col + patch_size // 2

#     # train_patch_mask[row1:(row2), col1] = 2
#     # train_patch_mask[row1:(row2 + 1), col2] = 2
#     # train_patch_mask[row1, col1:(col2)] = 2
#     # train_patch_mask[row2, col1:(col2 + 1)] = 2

#     if (df['set'][i] == 'train'):
#         train_patch_mask[row, col] = 5
#     elif (df['set'][i] == 'val'):
#         train_patch_mask[row, col] = 5
#     elif (df['set'][i] == 'test'):
#         test_patch_mask[row, col] = -5

# draw2(train_patch_mask + test_patch_mask, "patch_size_"+ str(patch_size) + "_2.png", "./", "RdBu", True)
# patch_mask = train_patch_mask + test_patch_mask
# print(str(len(patch_mask[patch_mask == 1])) + "\n" + 
# str(len(patch_mask[patch_mask == 2])) + "\n" + 
# str(len(patch_mask[patch_mask == 3])) + "\n" + 
# str(len(patch_mask[patch_mask == 3]) / (len(patch_mask[patch_mask == 2]) + len(patch_mask[patch_mask == 3])))
# .replace('.', '.'))
