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
from matplotlib import ticker, cm
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat
import torch

def aclr_fn(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
    """ 
        Calculate Adjacent Channel Leakage Ratio
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    ind1 = (nfft // 2) - int(np.ceil(nfft * f/fs))
    ind2 = (nfft // 2) + int(np.ceil(nfft * f/fs))
    aclr = np.sum(psd[ind1: ind2])/(np.sum(psd[:ind1]) + np.sum(psd[ind2:]))
    return 10 * np.log10(aclr)

dim = 10
figure = 2

tx_train = np.load('x.npy')
pa_out_train = np.load('d.npy')
# model_out_train = np.load(f'y_{dim}dim.npy')
model_out_train = np.load(f'y.npy')
tx_test = np.load('x_test.npy')
pa_out_test = np.load('d_test.npy')
# model_out_test = np.load(f'y_test_{dim}dim.npy')
model_out_test = np.load(f'y_test.npy')

f = 10 # MHz
fs = 245.76 # MHz
nfft = 1024

tap_num = 1
trans_len = 8

pa_powers = list(10 ** (np.load('pa_powers_round.npy') / 10))
# # pa_powers = list(np.arange(12))
# pa_powers = list(1 - np.arange(15.00, 6.00, -0.25)/15) + \
#             list(1 - np.arange(6.00, 2.00, -0.1)/15) + \
#             list(1 - np.arange(2, -0.25, -0.25)/15)
# pa_powers = pa_powers[:36] + pa_powers[36:77:10] + \
#             pa_powers[38:78:10] + pa_powers[41:81:10] + \
#             pa_powers[43:83:10] + pa_powers[77:]
# pa_powers = sorted(pa_powers)

# power_cases_train = [p for p in range(0, 43, 1)]
# power_cases_test = [p for p in range(0, 42, 1)]
power_cases_train = pa_powers[::2]
power_cases_test = pa_powers[1::2]
# power_cases_train = [p for p in range(0, 6, 1)]
# power_cases_test = [p for p in range(0, 6, 1)]


sig_len_train = int(len(tx_train) / len(power_cases_train))
sig_len_test = int(len(tx_test) / len(power_cases_test))

aclr_train = []
aclr_test = []

psd_train, psd_test, psd_train_nc, psd_test_nc, psd, psd_nc = [], [], [], [], [], []

for i in range(len(power_cases_train)):
    x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
    d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    y = model_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    nmse = sl.nmse(d, d - y)
    aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
    aclr_train.append(aclr_val)
    freqs, psd = sl.get_psd(x + d - y, Fs=fs, nfft=nfft)
    psd_train.append(psd)
    print(f"Case train {power_cases_train[i]:.2f} dB: ACLR = {aclr_val} dB")
    plt.figure(i + 1)
    plt.title(f"Power case {power_cases_train[i]} dB")
    pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    
for i in range(len(power_cases_test)):
    x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
    d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    y = model_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    nmse = sl.nmse(d, d - y)
    aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
    aclr_test.append(aclr_val)
    freqs, psd = sl.get_psd(x + d - y, Fs=fs, nfft=nfft)
    psd_test.append(psd)
    print(f"Case test {power_cases_test[i]:.2f} dB: ACLR = {aclr_val} dB")
    # plt.figure(i + 1)
    # plt.title(f"Power case {power_cases_test[i]} dB")
    # pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    
psd = list(np.zeros((len(pa_powers),)))
psd[::2] = psd_train
psd[1::2] = psd_test
psd = np.array(psd)
    
aclr_train_no_correct, aclr_test_no_correct = [], []
# Calculate ACLR for PA output without correction
for i in range(len(power_cases_train)):
    x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
    d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
    aclr_train_no_correct.append(aclr_val)
    freqs, psd_val = sl.get_psd(x + d, Fs=fs, nfft=nfft)
    psd_train_nc.append(psd_val)
    
for i in range(len(power_cases_test)):
    x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
    d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
    aclr_test_no_correct.append(aclr_val)
    freqs, psd_val = sl.get_psd(x + d, Fs=fs, nfft=nfft)
    psd_test_nc.append(psd_val)
    
psd_nc = list(np.zeros((len(pa_powers),)))
psd_nc[::2] = psd_train_nc
psd_nc[1::2] = psd_test_nc
psd_nc = np.array(psd_nc)

aclr_no_correct = list(np.zeros((len(pa_powers),)))
aclr_no_correct[::2] = aclr_train_no_correct
aclr_no_correct[1::2] = aclr_test_no_correct

plt.figure(figure)
# plt.plot(pa_powers[::2][::-1], [-p for p in aclr_train], marker='o')
# plt.plot(pa_powers[1::2][::-1], [-p for p in aclr_test], marker='o')
# plt.plot(pa_powers[::-1], [-p for p in aclr_no_correct], marker='o', color='black')
# plt.plot(pa_powers[::2], aclr_no_correct[::2], marker='o')
# plt.plot(pa_powers[1::2], aclr_no_correct[1::2], marker='o')
plt.xlabel('power inputs', fontsize=13)
plt.ylabel('ACLR, dB', fontsize=13)
# plt.yticks(np.arange(10, 50, 5))
# plt.legend(['train', 'test'], fontsize=13)
# plt.grid()

# psd_min = -89
# psd_max = -24
# psd[psd < psd_min] = psd_min
# # psd[psd >= -46] = -46
# psd_nc[psd_nc < psd_min] = psd_min
# # psd_nc[psd_nc >= -46] = -46

# # Draw PSD color map
# pa_range = np.arange(psd_min - 1, psd_max, 1)
# levels = list(pa_range)
# plt.figure(figure * 100)
# F, P = np.meshgrid(freqs, pa_powers)
# cs = plt.contourf(F, P, psd, levels=levels, cmap ="jet")
# cbar = plt.colorbar(cs) 
# plt.ylabel("PA power", fontsize=13)
# plt.xlabel("freq, MHz", fontsize=13)

# levels = list(pa_range)
# plt.figure(figure * 200)
# F, P = np.meshgrid(freqs, pa_powers)
# cs = plt.contourf(F, P, psd_nc, levels=levels, cmap ="jet")
# cbar = plt.colorbar(cs) 
# plt.ylabel("PA power", fontsize=13)
# plt.xlabel("freq, MHz", fontsize=13)

# w = torch.load(r'D:\Projects\МФТИ\En&T_2024\analyze\ten_dim_lin_scale\10_param_4_slot_61_cases_4_delay\weights_best_test_10_param_4_slot_61_cases_4_delay', map_location=torch.device('cpu'))
# w = torch.cat([p for j, p in enumerate(w.values()) if j < 1]).numpy()

# pl.plot_abs(w)