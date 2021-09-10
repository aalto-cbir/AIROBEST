import torch
import os
import numpy as np
from input.utils import split_data, get_patch

metadata = torch.load('./data/new_TAIGA/metadata.pt')
hyper_labels = torch.load('./data/new_TAIGA/hyperspectral_tgt_normalized.pt')

# use percentage of main tree species as mask if ignore_zero_labels is True
# => only care about forest areas
out_cls = metadata['num_classes']
idx = np.where(metadata['reg_label_names'] == 'leaf_area_index100')[0]
mask = hyper_labels[:, :, out_cls + idx]
print((mask != 0).sum().tolist())

R, C, _ = mask.size()

complete_image = []
for i in range(R):
    for j in range(C):
        if mask[i, j, 0] != 0:
            complete_image.append([i, j, 0, None, None])

print(len(complete_image))
np.save('./data/new_TAIGA/complete_image.npy', complete_image) 

complete_image_part1 = complete_image[0: 9000000]
complete_image_part2 = complete_image[9000000: 18000000]
complete_image_part3 = complete_image[18000000: len(complete_image)]


np.save('./data/new_TAIGA/complete_image1.npy', complete_image_part1)
np.save('./data/new_TAIGA/complete_image2.npy', complete_image_part2)
np.save('./data/new_TAIGA/complete_image3.npy', complete_image_part3)