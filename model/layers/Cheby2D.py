# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:13 2024

@author: Nikita Bakholdin, Alexander Degtyarev
"""
import torch
import torch.nn as nn
import sys

def tensor_memory_in_gb(tensor):
    bytes_used = tensor.element_size() * tensor.numel()
    gb_used = bytes_used / (1024 ** 3)
    return gb_used

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
        # input = torch.abs(input)
        input = 2 * torch.abs(input) - 1
        ind1 = torch.arange(self.order[0], device = self.device).to(input.dtype)
        ind2 = torch.arange(self.order[1], device = self.device).to(input.dtype)
        
        T0 = torch.cos(ind1[:, None] @ torch.arccos(input[:, :1, :])).permute(0, 2, 1)
        T1 = torch.cos(ind2[:, None] @ torch.arccos(input[:, 1:2, :])).permute(0, 2, 1)

        self.vand = (T0.unsqueeze(-1) * T1.unsqueeze(-2)).view(T0.shape[0], T0.shape[1], -1).to(self.dtype)

        approx = (self.vand @ self.weight)[:, None, :]
        return approx

    def get_jacobian(self, input):
        # input = torch.abs(input)
        input = 2 * torch.abs(input) - 1
        ind1 = torch.arange(self.order[0], device = self.device).to(input.dtype)
        ind2 = torch.arange(self.order[1], device = self.device).to(input.dtype)
        
        T0 = torch.cos(ind1[:, None] @ torch.arccos(input[:, :1, :])).permute(0, 2, 1)
        T1 = torch.cos(ind2[:, None] @ torch.arccos(input[:, 1:2, :])).permute(0, 2, 1)

        jacobian = (T0.unsqueeze(-1) * T1.unsqueeze(-2)).view(T0.shape[0], T0.shape[1], -1).to(self.dtype)[0, ...]
        return jacobian

class RrankCheby2D(nn.Module):
    """
        Introduces approximantion of rectangular 2D Chebyshev polynomial.
        Approximation is based on R-rank tensor decomposition.
    """
    def __init__(self, order=4, rank=1, dtype=torch.complex128, device='cuda:0'):
        super().__init__()
        assert type(order) == int or (type(order) == list and len(order) == 2), \
            "order parameter must be of an int type, or list including 2 ints."
        if type(order) == int:
            self.order = [order, order]
        else:
            self.order = order
        self.rank = rank
        self.dtype = dtype
        self.device = device

        self.nl1 = nn.ParameterList()
        self.nl2 = nn.ParameterList()
        for r in range(rank):
            self.nl1.append(torch.nn.Parameter(torch.zeros(self.order[0], dtype = dtype, device = device), requires_grad = True))
            self.nl2.append(torch.nn.Parameter(torch.zeros(self.order[1], dtype = dtype, device = device), requires_grad = True))
            self.nl1[r].data = 1.e-2 * (torch.rand(self.order[0], dtype = dtype, device = device) + 1j * torch.rand(self.order[0], dtype = dtype, device = device) - 1/2 - 1j/2)
            self.nl2[r].data = 1.e-2 * (torch.rand(self.order[1], dtype = dtype, device = device) + 1j * torch.rand(self.order[1], dtype = dtype, device = device) - 1/2 - 1j/2)
        
    def forward(self, input):
        input = torch.abs(input)
        ind1 = torch.arange(self.order[0], device = self.device).to(input.dtype)
        ind2 = torch.arange(self.order[1], device = self.device).to(input.dtype)
        
        T0 = torch.cos(ind1[:, None] @ torch.arccos(input[:, :1, :])).permute(0, 2, 1).to(self.dtype)
        T1 = torch.cos(ind2[:, None] @ torch.arccos(input[:, 1:2, :])).permute(0, 2, 1).to(self.dtype)
        # T0 = (ind1[:, None] @ input[:, :1, :]).permute(0, 2, 1).to(self.dtype)
        # T1 = (ind2[:, None] @ input[:, 1:2, :]).permute(0, 2, 1).to(self.dtype)

        approx = 0
        for r in range(self.rank):

            # T1_ = torch.roll(T1, shifts=self.rank - 2, dims=1)
            # T0_ = torch.roll(T0, shifts=self.rank - 2, dims=1)

            if r == 0:
                approx = ((T0 @ self.nl1[r]) * (T1 @ self.nl2[r]))[:, None, :]
            else:
                approx += ((T0 @ self.nl1[r]) * (T1 @ self.nl2[r]))[:, None, :]
        return approx

    def get_jacobian(self, input):
        input = torch.abs(input)
        ind1 = torch.arange(self.order[0], device = self.device).to(input.dtype)
        ind2 = torch.arange(self.order[1], device = self.device).to(input.dtype)
        
        T0 = torch.cos(ind1[:, None] @ torch.arccos(input[:, :1, :])).permute(0, 2, 1)[0, ...].to(self.dtype)
        T1 = torch.cos(ind2[:, None] @ torch.arccos(input[:, 1:2, :])).permute(0, 2, 1)[0, ...].to(self.dtype)

        jacobian = []
        for r in range(self.rank):

            # T1_ = torch.roll(T1, shifts=self.rank - 2, dims=0)
            # T0_ = torch.roll(T0, shifts=self.rank - 2, dims=0)

            jac_h0 = (T1 @ self.nl2[r])[:, None] * T0
            # jac_h0 = torch.diag(T1 @ self.nl2[r]) @ T0
            jacobian.append(jac_h0)

        for r in range(self.rank):

            # T1_ = torch.roll(T1, shifts=self.rank - 2, dims=0)
            # T0_ = torch.roll(T0, shifts=self.rank - 2, dims=0)

            jac_h1 = (T0 @ self.nl1[r])[:, None] * T1
            # jac_h1 = torch.diag(T0 @ self.nl1[r]) @ T1
            jacobian.append(jac_h1)

        jacobian = torch.cat(jacobian, dim=1)
        return jacobian