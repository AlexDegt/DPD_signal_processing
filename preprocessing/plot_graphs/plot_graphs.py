# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:00:18 2024

@author: ACER
"""

import sys, os
sys.path.insert(0, r'../lib')
import plot_lib as pl
import support_lib as sl
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat

# exp_name = "one_dim"
# exp_name = "three_dim"
# exp_name = "six_dim"
exp_name = "nine_dim"

power_cases = [-15, -12, -9, -6, -3, 0]

param_num = list(np.arange(4, 20, 4))
# param_num = list(np.arange(2, 22, 2))

nmse = np.load("nmse.npy")
aclr = np.load("aclr.npy")

assert nmse.shape == aclr.shape, "Shapes of nmse and aclr matrices must equal."
assert nmse.shape[0] == len(param_num)
assert nmse.shape[1] == len(power_cases)

plt.figure(1)
plt.grid()
legend = []
for i_param, param in enumerate(param_num):
    plt.plot(power_cases, aclr[i_param, :], marker='o')
    legend.append(f"{param}-param per NL")
plt.xticks(np.array(power_cases))
plt.ylabel("ACLR, dB", fontsize=13)
plt.xlabel("PA output power, dB", fontsize=13)
plt.legend(legend, fontsize=13, loc="upper right")   

plt.figure(2)
plt.grid()
legend = []
for i_param, param in enumerate(param_num):
    plt.plot(power_cases, nmse[i_param, :], marker='o')
    legend.append(f"{param}-param per NL")
plt.xticks(np.array(power_cases))
plt.ylabel("NMSE, dB", fontsize=13)
plt.xlabel("PA output power, dB", fontsize=13)
plt.legend(legend, fontsize=13, loc="upper right")  