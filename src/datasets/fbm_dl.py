from fbm import FBM
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def FBM_dl(num_samples, dim, length, hursts: float, batch_size: int,
           scale: float):
    fbm_paths = []
    for i in range(num_samples*dim):
        f = FBM(n=length, hurst=hursts, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(fbm_paths).reshape(
        num_samples, dim, length).permute(0, 2, 1)

    data = scale*data[:, 1:]
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


def FBM_data(num_samples, dim, length, h):
    fbm_paths = []
    for i in range(num_samples*dim):
        f = FBM(n=length, hurst=h, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(np.array(fbm_paths)).reshape(
        num_samples, dim, length+1).permute(0, 2, 1)
    return data
