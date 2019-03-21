import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


class ChenModel(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON CONVOLUTIONAL NEURAL NETWORKS
    Chen et al
    https://elib.dlr.de/106352/2/CNN.pdf
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.01)
            init.constant_(m.bias, 0)  # for 0.4.0 compatibility

    def __init__(self, input_channels, out_cls, out_reg, patch_size=27, n_planes=32):
        super(ChenModel, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 5, 5))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()
        print("Feature size:", self.features_size)

        self.fc_cls = nn.Linear(self.features_size, out_cls)
        self.fc_reg = nn.Linear(self.features_size, out_reg)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)

        # for classification task
        x_cls = F.sigmoid(self.fc_cls(x))

        # for regression task
        x_reg = self.fc_reg(x)

        return x_cls, x_reg


class LeeModel(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon (2016)
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7729859
    https://www.researchgate.net/publication/301876505_Contextual_Deep_CNN_Based_Hyperspectral_Classification
    https://arxiv.org/pdf/1604.03519.pdf
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            # init.kaiming_uniform_(m.weight)
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def __init__(self, in_channels, out_cls, out_reg):
        super(LeeModel, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, out_cls, (1, 1))
        self.conv9 = nn.Conv2d(128, out_reg, (1, 1))

        self.pool = nn.MaxPool2d((3, 3), padding=1, stride=1, dilation=1)

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)

        x_cls = self.conv8(x)
        x_cls = self.pool(x_cls)
        x_cls = F.sigmoid(x_cls)

        x_reg = self.conv9(x)
        x_reg = self.pool(x_reg)
        x_reg = F.sigmoid(x_reg)

        return x_cls, x_reg

class PhamModel(nn.Module):
    """
    CNN models for multi-task learning, inspired by Chen model
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def __init__(self, input_channels, out_cls, out_reg, metadata, patch_size=27, n_planes=32):
        super(PhamModel, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 5, 5))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()
        print("Feature size:", self.features_size)
        self.fc_shared = nn.Linear(self.features_size, 1024)

        categorical = metadata['categorical']
        self.n_cls = len(categorical.keys())
        self.n_reg = out_reg
        for idx, (key, values) in enumerate(categorical.items()):
            setattr(self, 'fc_cls_{}_1'.format(idx), torch.nn.Linear(1024, 200))
            setattr(self, 'fc_cls_{}_2'.format(idx), torch.nn.Linear(200, len(values)))

        for i in range(out_reg):
            setattr(self, 'fc_reg_{}_1'.format(i), torch.nn.Linear(1024, 200))
            setattr(self, 'fc_reg_{}_2'.format(i), torch.nn.Linear(200, 1))

        self.dropout = nn.Dropout(p=0.3)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def get_last_shared_layer(self):
        return self.fc_shared

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc_shared(x))

        if torch.cuda.is_available():
            pred_cls = torch.tensor([]).cuda()
        else:
            pred_cls = torch.tensor([])

        # for classification task
        for i in range(self.n_cls):
            layer1 = getattr(self, 'fc_cls_{}_1'.format(i))
            layer2 = getattr(self, 'fc_cls_{}_2'.format(i))
            x_cls = F.relu(layer1(x))
            pred_cls = torch.cat((pred_cls, F.softmax(layer2(x_cls))), 1)
            # pred_cls = torch.cat((pred_cls, layer2(x_cls)), 1)
        # for regression task
        if torch.cuda.is_available():
            pred_reg = torch.tensor([]).cuda()
        else:
            pred_reg = torch.tensor([])

        for i in range(self.n_reg):
            layer1 = getattr(self, 'fc_reg_{}_1'.format(i))
            layer2 = getattr(self, 'fc_reg_{}_2'.format(i))
            x_reg = F.relu(layer1(x))
            pred_reg = torch.cat((pred_reg, layer2(x_reg)), 1)
            # pred_reg = torch.cat((pred_reg, F.sigmoid(layer2(x_reg))), 1)  # performed worse

        return pred_cls, pred_reg


class ModelTrain(nn.Module):
    def __init__(self, model, criterion_cls_list, criterion_reg, metadata, options):
        super(ModelTrain, self).__init__()
        self.model = model
        self.task_count = model.n_cls + model.n_reg
        self.task_weights = torch.nn.Parameter(torch.tensor([1.0]*self.task_count).float())
        self.criterion_cls_list = criterion_cls_list
        self.criterion_reg = criterion_reg
        self.categorical = metadata['categorical']
        self.options = options

    def forward(self, src, tgt_cls, tgt_reg):
        task_loss = []
        pred_cls, pred_reg = self.model(src)
        start = 0

        for idx, (key, values) in enumerate(self.categorical.items()):
            n_classes = len(values)

            prediction, target = pred_cls[:, start:(start + n_classes)], tgt_cls[:, start:(start + n_classes)]
            # for cross entropy loss
            target = torch.argmax(target, dim=1).long()
            criterion_cls = self.criterion_cls_list[idx]

            if self.options.class_balancing == 'CRL' and self.model.training:
                single_loss = self.compute_objective_loss(src, target, prediction)
            else:
                single_loss = criterion_cls(prediction, target)

            task_loss.append(single_loss)
            start += n_classes

        for idx in range(self.model.n_reg):
            prediction, target = pred_reg[:, idx], tgt_reg[:, idx]
            single_loss = self.criterion_reg(prediction, target)
            task_loss.append(single_loss)

        return torch.stack(task_loss), pred_cls, pred_reg

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()

    def get_anchors(self, target):
        """
        Find the minor classes and assign all of the samples as anchors
        :return: an array contains indices of anchors
        """
        unique_values, unique_count = np.unique(target, return_counts=True)
        percentage = unique_count / np.sum(unique_count)
        sorted_percentage_idx = np.argsort(percentage)

        percentage_sum = 0
        minor_classes = np.array([], dtype='int64')
        for idx in sorted_percentage_idx:
            percentage_sum += percentage[idx]
            if percentage_sum < 0.5:
                minor_classes = np.append(minor_classes, idx)

        # minor_classes = torch.tensor(minor_classes, dtype=torch.int64, device=target.device)
        anchors = []

        if len(minor_classes) == 0:
            print('Target:', target)
            print('Percentage:', percentage)

        # sample_method = 'all'  # select anchors from all minor class
        # sample_method = 'equal'  # select certain amount of anchors from each minor class
        sample_method = 'weighted'

        if sample_method == 'all':
            for cls in minor_classes:
                # take all minor anchors
                anchors.append(np.argwhere(target == cls).flatten())
        elif sample_method == 'equal':
            count = 20
            for cls in minor_classes:
                # take all minor anchors
                anchors.append(np.argwhere(target == cls).flatten()[:count])
        elif sample_method == 'weighted':
            # minor_percentage = percentage[minor_classes]
            """
            # recompute percentage for minor classes only
            minor_percentage = minor_percentage / np.sum(minor_percentage)
            # taking the offset of minor_percentage as the percentage to select anchor samples
            minor_percentage = (1 - minor_percentage) / (len(minor_percentage) - 1)
            total_anchors = 100
            anchor_count = np.array(total_anchors * minor_percentage, dtype='int64')
            
            for cls, count in zip(minor_classes, anchor_count):
                # take all minor anchors
                anchors.append(np.argwhere(target == cls).flatten()[:count])
            """
            minor_percentage = 1 - percentage[minor_classes]
            for cls, percent in zip(minor_classes, minor_percentage):
                indices = np.argwhere(target == cls).flatten()
                count = int(percent * len(indices) + 1)
                anchors.append(indices[:count])
        return anchors, minor_classes

    def get_hard_positives(self, target, prediction, minor_class, kappa):
        """
        Hard negatives are data samples of a minority class c that have low prediction scores on class c by current
        model.
        :param target: true labels of the current batch, size: (batch_size x 1)
        :param prediction: prediction scores for each class of a single label, size: (batch_size x n_classes)
        :param minor_class: index of the minority class in consideration
        :param kappa: top k samples belong to class c that have lowest prediction scores on class c
        :return: indices of the hard-positive samples in the batch
        """
        class_samples_idx = np.argwhere(target == minor_class).flatten()
        topk_sorted_idx = np.argsort(prediction[class_samples_idx, minor_class])[:kappa]
        hard_positives = class_samples_idx[topk_sorted_idx]
        return hard_positives

    def get_hard_negatives(self, target, prediction, minor_class, kappa):
        """
        Hard nagatives are data samples of classes other than c, but have high prediction scores on class c by current
        model.
        :param target: true labels of the current batch, size: (batch_size x 1)
        :param prediction: prediction scores for each class of a single label, size: (batch_size x n_classes)
        :param minor_class: index of the minority class in consideration
        :param kappa: top k samples do not belong to class c but have highest prediction scores on class c
        :return: indices of the hard-positive samples in the batch
        """
        class_samples_idx = np.argwhere(target != minor_class).flatten()
        # get indices of 'kappa' wrong class predictions with highest probabilities
        bottomk_sorted_idx = np.argsort(prediction[class_samples_idx, minor_class])[-kappa:]
        hard_negatives = class_samples_idx[bottomk_sorted_idx]
        return hard_negatives

    def compute_class_rectification_loss(self, src, target, prediction, method='relative', level='class'):

        anchors, minor_classes = self.get_anchors(target)

        crl_loss = torch.tensor(0.0)
        T_size = 0

        if method == 'relative':
            for idx, minor_class in enumerate(minor_classes):
                hard_positives = self.get_hard_positives(target, prediction, minor_class, kappa=20)
                hard_negatives = self.get_hard_negatives(target, prediction, minor_class, kappa=20)
                T_size += len(anchors[idx]) * len(hard_positives) * len(hard_negatives)

                for a in anchors[idx]:
                    for p in hard_positives:
                        for n in hard_negatives:
                            if level == 'class':
                                mj = 0.5
                                dist_a_pos = abs(prediction[a, minor_class] - prediction[p, minor_class])
                                dist_a_neg = prediction[a, minor_class] - prediction[n, minor_class]
                                crl_loss += max(0, mj + dist_a_pos - dist_a_neg)
                            elif level == 'instance':
                                # TODO: implement
                                raise NotImplemented

            if T_size > 0:
                crl_loss = crl_loss / T_size
        elif method == 'absolute':
            # TODO: implement
            raise NotImplemented

        return crl_loss

    def compute_omega(self, target):
        unique_values, unique_count = np.unique(target, return_counts=True)
        if len(unique_values) == 1:
            return 0
        percentage = 100 * unique_count / np.sum(unique_count)
        percentage.sort()

        # omega_imb = percentage[-1] - percentage[-2]
        # omega_imb = percentage[-1] - percentage[0]
        # TODO: different methods to compute omega_imb: diff between most and least dominant classes, average diff, etc.
        omega_imb = (abs(percentage - np.mean(percentage)).sum() / len(percentage))
        return omega_imb

    def compute_objective_loss(self, src, target, prediction):

        eta = 0.01
        omega_imb = self.compute_omega(target)
        alpha = torch.tensor(eta * omega_imb, device=target.device)

        criterion = nn.CrossEntropyLoss()
        entropy_loss = criterion(prediction, target)

        target_npy = target.data.cpu().detach().numpy()
        prediction_npy = prediction.data.cpu().detach().numpy()
        crl_loss = self.compute_class_rectification_loss(src, target_npy, prediction_npy)
        crl_loss = crl_loss.to(target.device)

        loss = alpha * crl_loss + (1 - alpha) * entropy_loss

        return loss

