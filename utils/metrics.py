import torch
import numpy as np

# PyTorch implementation
def NMSE(x, y, desired):
    return 10.0 * torch.log10(
        (desired - y).abs().square().sum() /
        x.abs().square().sum())

# Numpy implementation
def nmse(x, e):
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e))/np.sum(x*np.conj(x)))))
    return y