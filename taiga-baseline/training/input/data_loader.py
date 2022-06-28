import torch
import torch.utils.data as data
import numpy as np
from input.utils import get_patch


class HypDataset(data.Dataset):
    """
    Custom dataset for hyperspectral data

    Description of data directory:
    - {train/test/val}.npy: contain pixel coordinates for each train/test/val set
    """

    def __init__(self, hyper_image, multiplier, hyper_labels_cls, hyper_labels_reg, coords, patch_size, model_name,
                 is_3d_convolution=False, augmentation=None):
        """

        :param hyper_image: hyperspectral image with shape WxHxC (C: number of channels)
        :param hyper_labels_cls: matrix of shape WxHxB (B: length of hyperspectral params)
        :param coords: array contains the list of training/validation indices (in form of (row, col)
        in image coordinates)
        :param hyperparams:
        """
        self.hyper_image = hyper_image
        self.multiplier = multiplier
        self.img_min, self.img_max = torch.min(hyper_image).float(), torch.max(hyper_image).float()
        #print('Max pixel %s, min pixel: %s' % (self.img_max, self.img_min))
        self.hyper_labels_cls = hyper_labels_cls
        self.hyper_labels_reg = hyper_labels_reg
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.is_3d_convolution = is_3d_convolution
        self.hyper_row = self.hyper_image.shape[0]
        self.hyper_col = self.hyper_image.shape[1]
        self.model_name = model_name
        # assert os.path.exists(coords_path), 'File does not exist in path: %s' % coords_path
        # self.coords = np.load(coords_path)
        self.coords = coords

    def idx2coord(self, idx):
        assert idx <= self.hyper_row * self.hyper_col, 'Invalid index in hyperspectral map'
        row = idx // self.hyper_row
        col = idx % self.hyper_col
        return row, col

    @staticmethod
    def flip(src):
        """
        Flip the tensor (hxwxb) randomly on vertical or horizontal axis
        :param src:
        :return:
        """
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            src = torch.flip(src, [0])
        if vertical:
            src = torch.flip(src, [1])
        return src

    @staticmethod
    def radiation_noise(src, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        # noise = torch.fromnumpy(np.random.normal(loc=0., scale=1.0, size=data.shape))
        noise = torch.normal(mean=0.0, std=torch.ones(src.shape))
        return alpha * src + beta * noise

    @staticmethod
    def mixture_noise(src1, src2, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = torch.normal(mean=0.0, std=torch.ones(src1.shape))
        return (alpha1 * src1 + alpha2 * src2) / (alpha1 + alpha2) + beta * noise

    def mixture_noise_deprecated(self, src, tgt_cls, tgt_reg, idx, beta=1 / 25):
        """
        Does not consider labels whose tensor rank !== 1
        :param src:
        :param tgt_cls:
        :param tgt_reg:
        :param idx:
        :param beta:
        :return:
        """
        if tgt_cls == float('inf') or tgt_reg == float('inf'):
            return src

        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        # noise = torch.fromnumpy(np.random.normal(loc=0., scale=1.0, size=data.shape))
        noise = torch.normal(mean=0.0, std=torch.ones(src.shape))
        src2 = None
        # search in the radius of 50 for data2
        radius = 50
        low = max(0, idx - radius)
        high = min(len(self.coords), idx + radius)
        for idx2 in range(low, high):
            if idx2 == idx:
                continue
            row2, col2 = self.coords[idx2]
            tgt_cls2 = self.hyper_labels_cls[row2, col2]
            if torch.all(torch.eq(tgt_cls, tgt_cls2)) == 0:
                continue
            tgt_reg2 = self.hyper_labels_reg[row2, col2]
            if torch.all(torch.eq(tgt_reg, tgt_reg2)) == 0:
                continue

            src2 = get_patch(self.hyper_image, row2, col2, self.patch_size).float()
            if self.multiplier is not None:
                src_norm_inv = get_patch(self.multiplier, row2, col2, self.patch_size)
                src_norm_inv = torch.unsqueeze(src_norm_inv, -1)
                src2 = src2 * src_norm_inv
            break

        if src2 is None:
            return alpha1 * src + beta * noise

        return (alpha1 * src + alpha2 * src2) / (alpha1 + alpha2) + beta * noise

    def __getitem__(self, idx):
        row, col, aug_code, row2, col2 = self.coords[idx]

        src = get_patch(self.hyper_image, row, col, self.patch_size).float()
        if self.multiplier is not None:
            src_norm_inv = get_patch(self.multiplier, row, col, self.patch_size)
            src_norm_inv = torch.unsqueeze(src_norm_inv, -1)
            src = src * src_norm_inv
        else:
            src = (src - self.img_min) / (self.img_max - self.img_min)
        if self.model_name == 'LeeModel':
            if self.hyper_labels_cls.nelement() == 0:
                tgt_cls = float('inf')
            else:
                tgt_cls = get_patch(self.hyper_labels_cls, row, col, self.patch_size)
                tgt_cls = tgt_cls.permute(2, 0, 1)
            if self.hyper_labels_reg.nelement() == 0:
                tgt_reg = float('inf')
            else:
                tgt_reg = get_patch(self.hyper_labels_reg, row, col, self.patch_size)
                tgt_reg = tgt_reg.permute(2, 0, 1)
        else:
            if self.hyper_labels_cls.nelement() == 0:
                tgt_cls = float('inf')
            else:
                tgt_cls = self.hyper_labels_cls[row, col]  # use labels of center pixel
            if self.hyper_labels_reg.nelement() == 0:
                tgt_reg = float('inf')
            else:
                tgt_reg = self.hyper_labels_reg[row, col]  # use labels of center pixel
        """
        if self.augmentation == 'flip' and self.patch_size > 1:
            src = self.flip(src)
        elif self.augmentation == 'radiation_noise' and np.random.random() < 0.1:
            src = self.radiation_noise(src)
        elif self.augmentation == 'mixture_noise' and np.random.random() < 0.2:
            src = self.mixture_noise(src, tgt_cls, tgt_reg, idx)
        """
        if aug_code == 1:
            src = torch.flip(src, [0])
        elif aug_code == 2:
            src = torch.flip(src, [1])
        elif aug_code == 3:
            src = self.radiation_noise(src)
        elif aug_code == 4:
            src2 = get_patch(self.hyper_image, row2, col2, self.patch_size).float()
            if self.multiplier is not None:
                src_norm_inv = get_patch(self.multiplier, row2, col2, self.patch_size)
                src_norm_inv = torch.unsqueeze(src_norm_inv, -1)
                src2 = src2 * src_norm_inv
            else:
                src2 = (src2 - self.img_min) / (self.img_max - self.img_min)
            src = self.mixture_noise(src, src2)

        # convert shape to pytorch image format: [channels x height x width]
        src = src.permute(2, 0, 1)

        # Transform to 4D tensor for 3D convolution
        if self.is_3d_convolution and self.patch_size > 1:
            src = src.unsqueeze(0)
        return src, tgt_cls, tgt_reg, idx

    def __len__(self):
        return len(self.coords)


def get_loader(hyper_image, multiplier, hyper_labels_cls, hyper_labels_reg, coords, batch_size, patch_size=11,
               model_name='ChenModel', shuffle=False, num_workers=0, is_3d_convolution=False, augmentation=None, pin_memory=False):

    dataset = HypDataset(hyper_image, multiplier, hyper_labels_cls, hyper_labels_reg, coords, patch_size,
                         model_name=model_name, is_3d_convolution=is_3d_convolution, augmentation=augmentation)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
