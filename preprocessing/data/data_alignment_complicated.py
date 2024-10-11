# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:02:52 2023

@author: dWX1065688
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

# Modified numpy.ndarray
class modndarray(np.ndarray):
    # Hermitian conjugation
    def herm(x):
        return np.conj(x).T

save_path = "D:\Projects\МФТИ\En&T_2024\data"
names1, names2, names3 = [], [], []
for p in np.arange(-15, -6, 0.25):
    int_part = str(int(abs(p)))
    frac_part = str(int(abs(p) % 1 * 100))
    if int_part == '0':
        int_part = '00'
    elif int_part != '0' and len(int_part) == 1:
        int_part = '0' + int_part
    if len(frac_part) == 1:
        frac_part = frac_part + '0'
    names1.append('m' + int_part + '_' + frac_part)
for p in np.arange(-6, -2, 0.1):
    int_part = str(int(abs(p)))
    frac_part = str(int(abs(p) % 1 * 100))
    if int_part == '0':
        int_part = '00'
    elif int_part != '0' and len(int_part) == 1:
        int_part = '0' + int_part
    if len(frac_part) == 1:
        frac_part = frac_part + '0'
    names2.append('m' + int_part + '_' + frac_part)
for p in np.arange(-2, 0.25, 0.25):
    int_part = str(int(abs(p)))
    frac_part = str(int(abs(p) % 1 * 100))
    if int_part == '0':
        int_part = '00'
    elif int_part != '0' and len(int_part) == 1:
        int_part = '0' + int_part
    if len(frac_part) == 1:
        frac_part = frac_part + '0'
    names3.append('m' + int_part + '_' + frac_part)
names = names1 + names2 + names3
names = names[::-1]

RX_datas = np.load('RX_datas_025dB_step.npy').view(modndarray)
# RX_datas = np.load('RX_datas.npy').view(modndarray)
# RX_datas = np.load('RX_datas_half_range.npy').view(modndarray)
x = loadmat('Tx_20MHz_Fs245p76.mat')['x'].reshape(-1)[:RX_datas.shape[1]].view(modndarray)

case_num = RX_datas.shape[0]

tap_num = 1

# alpha_real = [30000, 34500, 40000, 48000, 62000, 85000]
# alpha_real = [34000, 39500, 46500, 57500, 73687]
# alpha_real = [30000, 31500, 33000, 34500, 37000, 38500, 41000, 43000, 46000, 49000, 54000, 58000, 62000, 68000, 74000, 85000]
alpha_real = [30000, 30700, 31200, 31600, 32000, 32400, 32800, 33200, 33600] + \
              list(np.arange(33780, 33780 + 40 * 180, 180)) + \
              list(np.arange(41500, 41500 + 12 * 600, 600)) + \
               list(np.arange(50500, 50500 + 10 * 1000, 1000)) + \
               list(np.arange(60500, 60500 + 10 * 1400, 1400)) + \
               list(np.arange(77000, 77000 + 4 * 2200, 2200))
              
for i in range(case_num):
    # Firstly we find real gain for x
    x_scaled = x / alpha_real[i]
    # pl.plot_psd(x_scaled, nfig=1, clf=False)
    # pl.plot_psd(RX_datas[i, :], nfig=1, clf=False)
    # pl.plot_psd(x_scaled, RX_datas[i, :], nfig=1, clf=False)
    U = sl.fir_matrix_generate(RX_datas[i, :], tap_num).view(modndarray)
    hess = U.herm() @ U
    reg = 0 # np.max(np.abs(hess)) * 1e-3 * np.eye(tap_num)
    w = np.linalg.pinv(hess + reg) @ U.herm() @ x_scaled
    pa_out = U @ w
    # pl.plot_psd(x_scaled, pa_out, pa_out - x_scaled)
    mat = {'TX': x_scaled, 'PAout': pa_out}
    savemat(os.path.join(save_path, f'aligned_{names[i]}dB_100RB_Fs245p76.mat'), mat)