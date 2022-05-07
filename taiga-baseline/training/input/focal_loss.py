import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25, weight=None):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.weight = weight

        print('Weight:', weight)

    def forward(self, prediction, target):
        log_pt = - F.cross_entropy(prediction, target, weight=self.weight)
        pt = torch.exp(log_pt)

        focal_loss = -((1 - pt) ** self.focusing_param) * log_pt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss.sum()
