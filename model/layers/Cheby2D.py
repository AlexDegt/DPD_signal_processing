# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:13 2024

@author: Nikita
"""
import torch
import torch.nn as nn


class Cheby2D(nn.Module):
    def __init__(self, delay = 0, order = 4, complex_coef = True, uniCalc = False, dtype = torch.complex128, device = 'cuda'):
        super().__init__()
        self.order = order
        self.dtype = dtype
        self.device = device
        self.vand = None
        self.Cheby2D = torch.nn.Parameter(torch.zeros(order**2, dtype = dtype, device = device), requires_grad = True)
        self.Cheby2D.data = 1.e-2 * (torch.rand(order**2, dtype = dtype, device = device) + 1j * torch.rand(order**2, dtype = dtype, device = device) - 1/2 - 1j/2)
        
    def forward(self, input):
        input = torch.abs()
        ind = torch.arange(self.order, device = self.device)
        
        T0 = torch.cos(ind[:, None] * torch.arccos(input[0, :1, :]))
        T1 = torch.cos(ind[:, None] * torch.arccos(input[0, 1:2, :]))
        
        self.vand = (T0[:, None, :] * T1[None, :, :]).reshape(-1, T0.shape[-1]).T.to(self.dtype)
        
        approx = (self.vand @ self.Cheby2D)[None, None, :]
        return approx
    
class Cheby2D_delay(nn.Module):
    def __init__(self, delay, order = 4, complex_coef = True, uniCalc = False, dtype = torch.complex128, device = 'cuda'):
        super().__init__()
        self.delay = Delay(delay)
        self.Cheby = Cheby2D()
        
    def forward(self, input):
        out = self.delay(input)
        return self.Cheby(out)
       
    
    "Rewrite the Delay function"
class Delay(nn.Module):
    def __init__(self, M):
        super(Delay, self).__init__()
        self.M = M
        self.op = nn.Sequential(nn.ConstantPad1d(abs(M),0))

    def forward(self, x):
        return self.op(x)[:, :, :x.shape[2]] if self.M > 0 else self.op(x)[:, :, -x.shape[2]:]
    
    
# class Delay(nn.Module):
#     def __init__(self, delays):
#         super(Delay, self).__init__()
#         self.delays = delays

# "Check delay"
# device = 'cuda'
# x = torch.tensor([5, 1, -3, 2, -8, 2, -3, 7, -4, 0]).reshape(1, 1, -1).to(device)
# delay = Delay(-5)
# print(delay(x))
