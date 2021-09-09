import torch
import numpy as np

targets = torch.load("./data/TAIGA/hyperspectral_tgt_normalized.pt")
print('targets', targets.size())

for i in range(20):
    target = targets[:, :, i]
    print(i, np.where(target.flatten() != 0)[0].shape)
