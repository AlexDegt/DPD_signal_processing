# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:56:53 2024

@author: Nikita
"""
import torch
import torch.nn as nn

from .layers import Delay, Cheby2D

# class Cheby_parallel_2D(nn.Module):
#     def __init__ (self, delays):
#         super(Cheby_parallel_2D, self).__init__()
        
#         self.delays = delays
#         # self.delay =` Delay(delays)
#         self.cells = nn.ModuleList()
#         for i in range(len(delays)):
#             # print(i)
#             self.cells.append(Cheby2D())
    
#     def forward (self, x):
#         out = torch.zeros_like(x)
        
#         for i in range(len(self.cells)):
#             delay = Delay(self.delays[i])
#             out += self.cells[i](delay(x))
            
#         return out

class Cheby_parallel_2D(nn.Module):
    def __init__ (self, delays):
        super(Cheby_parallel_2D, self).__init__()
        
        self.cells = nn.ModuleList()
        for i in range(len(delays)):
            # print(i)
            self.cells.append(Cheby2D())
    
    def forward (self, x):
        out = torch.zeros_like(x)
        
        for i in range(len(self.cells)):
            delay = Delay(self.delays[i])
            out += self.cells[i](delay(x))
            
        return out