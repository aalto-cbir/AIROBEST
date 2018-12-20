import torch
import torch.utils.data as data
from input.utils import get_patch


class HypDataset(data.Dataset):
    """
    Custom dataset for hyperspectral data

    Description of data directory:
    - {train/test/val}.npy: contain pixel coordinates for each train/test/val set
    """

    def __init__(self, hyper_image, norm_inv, hyper_labels_cls, hyper_labels_reg, coords, patch_size, model_name,
                 is_3d_convolution=False, device='cpu'):
        """

        :param hyper_image: hyperspectral image with shape WxHxC (C: number of channels)
        :param hyper_labels_cls: matrix of shape WxHxB (B: length of hyperspectral params)
        :param coords: array contains the list of training/validation indices (in form of (row, col)
        in image coordinates)
        :param hyperparams:
        """
        self.hyper_image = hyper_image
        self.norm_inv = norm_inv
        self.hyper_labels_cls = hyper_labels_cls
        self.hyper_labels_reg = hyper_labels_reg
        self.patch_size = patch_size
        self.is_3d_convolution = is_3d_convolution
        self.hyper_row = self.hyper_image.shape[0]
        self.hyper_col = self.hyper_image.shape[1]
        self.model_name = model_name
        self.device = device
        # assert os.path.exists(coords_path), 'File does not exist in path: %s' % coords_path
        # self.coords = np.load(coords_path)
        self.coords = coords

    def idx2coord(self, idx):
        assert idx <= self.hyper_row * self.hyper_col, 'Invalid index in hyperspectral map'
        row = idx // self.hyper_row
        col = idx % self.hyper_col
        return row, col

    def __getitem__(self, idx):
        row, col = self.coords[idx]

        src = get_patch(self.hyper_image, row, col, self.patch_size).to(self.device, dtype=torch.float32)
        src_norm_inv = get_patch(self.norm_inv, row, col, self.patch_size).to(self.device, dtype=torch.float32)
        src_norm_inv = torch.unsqueeze(src_norm_inv, -1)
        src = src * src_norm_inv
        if self.model_name == 'LeeModel':
            tgt_cls = get_patch(self.hyper_labels_cls, row, col, self.patch_size)
            tgt_cls = tgt_cls.permute(2, 0, 1)
            tgt_reg = get_patch(self.hyper_labels_reg, row, col, self.patch_size)
            tgt_reg = tgt_reg.permute(2, 0, 1)
        else:
            tgt_cls = self.hyper_labels_cls[row, col]  # use labels of center pixel
            tgt_reg = self.hyper_labels_reg[row, col]  # use labels of center pixel

        # convert shape to pytorch image format: [channels x height x width]
        src = src.permute(2, 0, 1)

        # Transform to 4D tensor for 3D convolution
        if self.is_3d_convolution and self.patch_size > 1:
            src = src.unsqueeze(0)
        return src, tgt_cls, tgt_reg

    def __len__(self):
        return len(self.coords)


def get_loader(hyper_image, norm_inv, hyper_labels_cls, hyper_labels_reg, coords, batch_size, patch_size=11,
               model_name='ChenModel', device=None, shuffle=False, num_workers=0, is_3d_convolution=False):
    dataset = HypDataset(hyper_image, norm_inv, hyper_labels_cls, hyper_labels_reg, coords, patch_size,
                         model_name=model_name, is_3d_convolution=is_3d_convolution, device=device)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return data_loader