# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:13 2024

@author: Nikita
"""
import torch
import torch.nn as nn
import sys

class Cheby2D(nn.Module):
    """
        Introduces rectangular 2D Chebyshev polynomial.
    """
    def __init__(self, order=4, dtype=torch.complex128, device='cuda:0'):
        super().__init__()
        assert type(order) == int or (type(order) == list and len(order) == 2), \
            "order parameter must be of an int type, or list including 2 ints."
        if type(order) == int:
            self.order = [order, order]
        else:
            self.order = order
        self.dtype = dtype
        self.device = device
        self.vand = None
        param_num = self.order[0] * self.order[1]
        self.weight = torch.nn.Parameter(torch.zeros(param_num, dtype = dtype, device = device), requires_grad = True)
        self.weight.data = 1.e-2 * (torch.rand(param_num, dtype = dtype, device = device) + 1j * torch.rand(param_num, dtype = dtype, device = device) - 1/2 - 1j/2)
        
    def forward(self, input):
        input = torch.abs(input)
        ind1 = torch.arange(self.order[0], device = self.device)
        ind2 = torch.arange(self.order[1], device = self.device)
        
        T0 = torch.cos(ind1[:, None] * torch.arccos(input[0, :1, :]))
        T1 = torch.cos(ind2[:, None] * torch.arccos(input[0, 1:2, :]))
        
        self.vand = (T0[:, None, :] * T1[None, :, :]).reshape(-1, T0.shape[-1]).T.to(self.dtype)

        approx = (self.vand @ self.weight)[None, None, :]
        return approx