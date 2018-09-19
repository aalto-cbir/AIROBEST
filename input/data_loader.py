import torch
import torch.utils.data as data
import os


class HypDataset(data.Dataset):
    """
    Custom dataset for hyperspectral data

    Description of data directory:
    - src_{image_id}.npy
    - tgt_{image_id}.npy
    - {train/test/val}.txt: contain image ids for each train/test/val set
    """
    def __init__(self, root_dir, dataset_name):
        self.root_dir = root_dir
        self.features = []
        self.target = []

        with open(root_dir + dataset_name) as f:
            for id in f:
                self.features.append('{}/src_{}.npy'.format(root_dir, id))
                self.target.append('{}/tgt_{}.npy'.format(root_dir, id))

    def __getitem__(self, index):
        image_id = f'{index:10}'  # pad 0 in front of image id
        src_path = '{}/src_{}.npy'.format(self.root_dir, image_id)
        tgt_path = '{}/tgt_{}.npy'.format(self.root_dir, image_id)
        assert os.path.exists(src_path), \
            'Source file not found in %s ' % src_path
        assert os.path.exists(tgt_path), \
            'Target file not found in %s ' % tgt_path

        src = torch.load(src_path)
        tgt = torch.load(tgt_path)
        return src, tgt

    def __len__(self):
        return len(self.features)


def get_loader(root_dir, dataset_name, batch_size, shuffle, num_workers):
    dataset = HypDataset(root_dir, dataset_name)

    # TODO: collate_fn?
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return data_loader