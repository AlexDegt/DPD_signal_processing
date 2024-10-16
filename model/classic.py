# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:56:53 2024

@author: Nikita
"""
import torch
import torch.nn as nn
import sys

from .layers import Cheby2D, Delay

class ParallelCheby2D(nn.Module):
    def __init__ (self, order, delays, dtype=torch.complex128, device='cuda:0'):
        super(ParallelCheby2D, self).__init__()
        
        self.dtype = dtype
        self.device = device
        # Must be list of 2 ints or 1 int
        self.order = order
        delays_input = [delays_branch[1:] for delays_branch in delays]
        delays_output = [delays_branch[:1] for delays_branch in delays]
        self.delay_inp = Delay(delays_input, dtype, device)
        self.delay_out = Delay(delays_output, dtype, device)
        self.cells = nn.ModuleList()
        self.trans_len = int(len(delays) // 2)
        for i in range(len(delays)):
            self.cells.append(Cheby2D(order, dtype, device))
    
    def forward(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)
        output = sum([x_in[:, j_branch, ...] * cell(x_curr[:, j_branch, ...]) for j_branch, cell in enumerate(self.cells)])
        return output[..., self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None]