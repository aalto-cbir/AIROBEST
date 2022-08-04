import torch
import argparse
import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split


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
    random_state = 797  # 646, 919, 390, 101
    R, C, _ = mask.shape
    if mode == 'grid':
        stride = patch_size
    for i in range(patch_size // 2, rows - patch_size // 2, stride):
        for j in range(patch_size // 2, cols - patch_size // 2, stride):
            patch = get_patch(mask, i, j, patch_size)
            if torch.min(patch) > 0:  # make sure there is no black pixels in the patch
                if mode == 'random' or mode == 'grid':
                    coords.append([i, j, 0, None, None])
    if mode == 'random' or mode == 'grid':
        print('Random seed:', random_state)
        train, val = train_test_split(coords, train_size=0.9, random_state=random_state, shuffle=True)

    new_data = []
    #for sample in train:
    #    r, c, _, _, _ = sample
    #        # if aug == 'flip':
    #            # code = 1 if np.random.random() > 0.5 else 2
    #    code = 1
    #    new_data.append([r, c, code, None, None])
    #
    #train = train + new_data
    np.random.seed(123)
    np.random.shuffle(train)
    train = np.array(train)
    val = np.array(val)
    coords = np.array(coords)
    print('Number of training pixels: %d, val pixels: %d' % (len(train), len(val)))
    # print('Train', train[0:10])
    # print('Val', val[0:10])
    return train, test, val, coords


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-patch_size', type=int,
                        required=True, default=45,
                        help="Original patch size")
    parser.add_argument('-new_patch_size', type=int,
                        required=True, default=13,
                        help="New patch size")
    parser.add_argument('-data_path', type=str, 
                        required=True, default='',
                        help="Path to data")
    parser.add_argument('-save_path', type=str, 
                        required=True, default='',
                        help="Path to save outputs")

    opt = parser.parse_args()

    return opt 

def main():
    infer_opts = parse_args()

    patch_size = infer_opts.patch_size
    new_patch_size = infer_opts.new_patch_size
    path = infer_opts.data_path
    save_path = infer_opts.save_path

    metadata = torch.load(path + '/metadata.pt')
    hyper_labels = torch.load(path + '/hyperspectral_tgt_normalized.pt')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Split data patch size 27 => (optional, add val + test in previous split to mask)
    # 2. Add to mask
    # 3. Split data size 19
    # 4. Double check train_set to remove overlapping points with mask

    # use percentage of main tree species as mask if ignore_zero_labels is True
    # => only care about forest areas
    out_cls = metadata['num_classes']
    idx = np.where(metadata['reg_label_names'] == 'leaf_area_index100')[0]
    mask = hyper_labels[:, :, out_cls + idx]
    print("a: ", (mask != 0).sum().tolist(), "idx:", idx, out_cls)
    np.save(path + '/mask.npy', mask)

    R, C, _ = mask.size()

    train_mask, test_mask, val_mask, origin_mask = split_data(R, C, mask, [], [], patch_size, patch_size, 'grid', False)
    print('Val mask: ', len(val_mask))
    print('Test mask: ', len(test_mask))

    for i in range(len(val_mask)):
        mask[val_mask[i][0] - patch_size // 2 : val_mask[i][0] + patch_size // 2 + 1, val_mask[i][1] - patch_size // 2 : val_mask[i][1] + patch_size // 2 + 1] = 0

    for i in range(len(test_mask)):
        mask[test_mask[i][0] - patch_size // 2 : test_mask[i][0] + patch_size // 2 + 1, test_mask[i][1] - patch_size // 2 : test_mask[i][1] + patch_size // 2 + 1] = 0

    print("b: ", (mask != 0).sum().tolist())


    train_set, test_set, val_set, origin_set = split_data(R, C, mask, [], [], new_patch_size, new_patch_size, 'grid', True)

    np.save(save_path + '/train_set.npy', train_set)
    np.save(save_path + '/test_set.npy', test_mask)
    np.save(save_path + '/val_set.npy', val_mask)
    np.save(save_path + '/origin_set.npy', origin_set)

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

    np.save(save_path + '/train_set.npy', new_train_set)
    print('New train set: ', len(new_train_set))

if __name__ == "__main__":
    main()
