import numpy as np
import torch
import torch.utils.data as data
import os
import input.utils as utils


class HypDataset(data.Dataset):
    """
    Custom dataset for hyperspectral data

    Description of data directory:
    - {train/test/val}.npy: contain pixel coordinates for each train/test/val set
    """

    def __init__(self, hyper_image, hyper_data, coords_path, **hyperparams):
        """

        :param hyper_image: hyperspectral image with shape WxHxC (C: number of channels)
        :param hyper_data: matrix of shape WxHxB (B: length of hyperspectral params)
        :param coords_path: path to the file contains the list of training/validation indices (in form of (row, col)
        in image coordinates)
        :param hyperparams:
        """
        self.hyper_image = hyper_image
        self.hyperdata = hyper_data
        self.patch_size = hyperparams['patch_size']
        self.hyper_row = self.hyper_image.shape(1)
        self.hyper_col = self.hyper_image.shape(0)
        assert os.path.exists(coords_path), 'File does not exist in path: %s' % coords_path
        self.coords = np.load(coords_path)

    def idx2coord(self, idx):
        assert idx <= self.hyper_row * self.hyper_col, 'Invalid index in hyperspectral map'
        row = idx // self.hyper_row
        col = idx % self.hyper_col
        return row, col

    def __getitem__(self, idx):
        row, col = self.coords[idx]
        row1, col1 = row - self.patch_size // 2, col - self.patch_size // 2
        row2, col2 = row + self.patch_size // 2, col + self.patch_size // 2

        assert row1 >= 0 and col1 >= 0 and row2 <= self.hyper_row and col2 <= self.hyper_col, \
            'Coordinate is invalid: %s %s ' % (row, col)

        src = self.hyper_image[row1:row2, col1:col2]
        tgt = self.hyperdata[row1:row2, col1:col2]

        # make sure to return torch tensors
        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)

        return src, tgt

    def __len__(self):
        return len(self.coords)


def get_loader(hyper_image, hyper_data, coords_path, batch_size, shuffle, num_workers, **hyperparams):
    dataset = HypDataset(hyper_image, hyper_data, coords_path, **hyperparams)

    # TODO: collate_fn?
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return data_loader