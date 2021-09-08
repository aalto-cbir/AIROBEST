import torch
import os
import numpy as np
from input.utils import split_data, get_patch

patch_size = 45
new_patch_size = 15
path = './data/TAIGA/test-splits-split45'

metadata = torch.load('./data/TAIGA/metadata.pt')
hyper_labels = torch.load('./data/TAIGA/hyperspectral_tgt_normalized.pt')

# 1. Split data patch size 27 => (optional, add val + test in previous split to mask)
# 2. Add to mask
# 3. Split data size 19
# 4. Double check train_set to remove overlapping points with mask

# use percentage of main tree species as mask if ignore_zero_labels is True
# => only care about forest areas
out_cls = metadata['num_classes']
idx = np.where(metadata['reg_label_names'] == 'leaf_area_index100')[0]
mask = hyper_labels[:, :, out_cls + idx]
print((mask != 0).sum().tolist())

R, C, _ = mask.size()

train_mask, test_mask, val_mask, origin_mask = split_data(R, C, mask, [], [], patch_size, patch_size, 'grid', False)
print('Val mask: ', len(val_mask))
print('Test mask: ', len(test_mask))

for i in range(len(val_mask)):
    mask[val_mask[i][0] - patch_size // 2 : val_mask[i][0] + patch_size // 2 + 1, val_mask[i][1] - patch_size // 2 : val_mask[i][1] + patch_size // 2 + 1] = 0

for i in range(len(test_mask)):
    mask[test_mask[i][0] - patch_size // 2 : test_mask[i][0] + patch_size // 2 + 1, test_mask[i][1] - patch_size // 2 : test_mask[i][1] + patch_size // 2 + 1] = 0

print((mask != 0).sum().tolist())


train_set, test_set, val_set, origin_set = split_data(R, C, mask, [], [], new_patch_size, new_patch_size, 'grid', True)

np.save(path + '/train_set.npy', train_set)
np.save(path + '/test_set.npy', test_mask)
np.save(path + '/val_set.npy', val_mask)
np.save(path + '/origin_set.npy', origin_set)

print('Train set: ', len(train_set))
print('Test set: ', len(test_set))
print('Val set: ', len(val_set))
print('Origin set: ', len(origin_set))

index_to_be_deleted = []

for i in range (len(train_set)):
    patch = get_patch(mask, train_set[i][0], train_set[i][1], patch_size)
    if torch.min(patch) == 0:
        index_to_be_deleted.append(i)

print(len(index_to_be_deleted))

new_train_set = np.delete(train_set, index_to_be_deleted, 0)

np.save(path + '/train_set.npy', new_train_set)
print('New train set: ', len(new_train_set))