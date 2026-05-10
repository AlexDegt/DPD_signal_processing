# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:56:53 2024

@author: Nikita
"""
import torch
import torch.nn as nn
import sys

from .layers import Cheby2D, RrankCheby2D, LinearSpline2D, Delay

def tensor_memory_in_gb(tensor):
    bytes_used = tensor.element_size() * tensor.numel()
    gb_used = bytes_used / (1024 ** 3)
    return gb_used

class ParallelSpline2D(nn.Module):
    """
        Class represent 2-dimensional non-linearity based on linear-spline polynomials
    """
    def __init__ (self, order, delays, dtype=torch.complex128, device='cuda:0'):
        super(ParallelSpline2D, self).__init__()
        
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
            self.cells.append(LinearSpline2D(order, dtype, device))
    
    def forward(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)
        output = sum([x_in[:, j_branch, ...] * cell(x_curr[:, j_branch, ...]) for j_branch, cell in enumerate(self.cells)])
        return output[..., self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None]

    def get_jacobian(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)

        jacobian = []
        for j_branch, cell in enumerate(self.cells):
            jac_curr = x_in[0, j_branch, 0, ...][:, None] * cell.get_jacobian(x_curr[:, j_branch, ...])
            # jac_curr = torch.diag(x_in[0, j_branch, 0, ...]) @ cell.get_jacobian(x_curr[:, j_branch, ...])
            jac_curr = jac_curr[self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None, :]
            jacobian.append(jac_curr)
        jacobian = torch.cat(jacobian, dim=1)
        return jacobian

    def get_flat_params(self):
        flat_params = []
        for cell in self.cells:
            flat_params.append(cell.weight.view(-1))
        flat_params = torch.cat(flat_params, dim=0)
        return flat_params

    def set_flat_params(self, flat_params):
        offset = 0
        for j_branch, cell in enumerate(self.cells):
            numel = cell.weight.numel()
            cell.weight.copy_(flat_params[offset:offset + numel].view_as(cell.weight))
            offset += numel

class ParallelCheby2D(nn.Module):
    """
        Class represent 2-dimensional non-linearity based on Chebyshev polynomials
    """
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

    def get_jacobian(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)

        jacobian = []
        for j_branch, cell in enumerate(self.cells):
            jac_curr = x_in[0, j_branch, 0, ...][:, None] * cell.get_jacobian(x_curr[:, j_branch, ...])
            # jac_curr = torch.diag(x_in[0, j_branch, 0, ...]) @ cell.get_jacobian(x_curr[:, j_branch, ...])
            jac_curr = jac_curr[self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None, :]
            jacobian.append(jac_curr)
        jacobian = torch.cat(jacobian, dim=1)
        return jacobian

    def get_flat_params(self):
        flat_params = []
        for cell in self.cells:
            flat_params.append(cell.weight.view(-1))
        flat_params = torch.cat(flat_params, dim=0)
        return flat_params

    def set_flat_params(self, flat_params):
        offset = 0
        for j_branch, cell in enumerate(self.cells):
            numel = cell.weight.numel()
            cell.weight.copy_(flat_params[offset:offset + numel].view_as(cell.weight))
            offset += numel

class ParallelRrankCheby2D(nn.Module):
    """
        Class represent R-rank approximation of 2-dimensional non-linearity based on Chebyshev polynomials.
        Approximation is based on R-rank tensor decomposition.
    """
    def __init__ (self, order, rank, delays, dtype=torch.complex128, device='cuda:0'):
        super(ParallelRrankCheby2D, self).__init__()
        
        self.dtype = dtype
        self.device = device
        # Must be list of 2 ints or 1 int
        self.order = order
        self.rank = rank
        delays_input = [delays_branch[1:] for delays_branch in delays]
        delays_output = [delays_branch[:1] for delays_branch in delays]
        self.delay_inp = Delay(delays_input, dtype, device)
        self.delay_out = Delay(delays_output, dtype, device)
        self.cells = nn.ModuleList()
        self.trans_len = int(len(delays) // 2)
        for i in range(len(delays)):
            self.cells.append(RrankCheby2D(order, rank, dtype, device))
    
    def forward(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)
        output = sum([x_in[:, j_branch, ...] * cell(x_curr[:, j_branch, ...]) for j_branch, cell in enumerate(self.cells)])
        return output[..., self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None]

    def get_jacobian(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)

        jacobian = []
        for j_branch, cell in enumerate(self.cells):
            jac_curr = x_in[0, j_branch, 0, ...][:, None] * cell.get_jacobian(x_curr[:, j_branch, ...])
            # jac_curr = torch.diag(x_in[0, j_branch, 0, ...]) @ cell.get_jacobian(x_curr[:, j_branch, ...])
            jac_curr = jac_curr[self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None, :]
            jacobian.append(jac_curr)
        jacobian = torch.cat(jacobian, dim=1)
        return jacobian

    def get_flat_params(self):
        flat_params = []
        for cell in self.cells:
            for nl in cell.nl1:
                flat_params.append(nl.view(-1))
            for nl in cell.nl2:
                flat_params.append(nl.view(-1))
        flat_params = torch.cat(flat_params, dim=0)
        return flat_params

    def set_flat_params(self, flat_params):
        offset = 0
        for j_branch, cell in enumerate(self.cells):
            for nl in cell.nl1:
                numel = nl.numel()
                nl.copy_(flat_params[offset:offset + numel].view_as(nl))
                offset += numel
            for nl in cell.nl2:
                numel = nl.numel()
                nl.copy_(flat_params[offset:offset + numel].view_as(nl))
                offset += numel