# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:02:52 2023

@author: dWX1065688
"""
import sys, os
sys.path.insert(0, r'../../lib')
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
    guard = 4 # 10
    aclr = np.sum(psd[ind2 + guard: 2 * ind2 - ind1 + guard])/np.sum(psd[ind1: ind2])
    return 10 * np.log10(aclr)

# Determine powers for model input for train and test correspondingly
pa_powers = list(10 ** ((np.load('pa_powers_round.npy') - 1)/ 10))
power_cases_train = pa_powers[::2]
power_cases_test = pa_powers[1::2]

# Determine parameters of the simulation, which is chosen for PSD and ACLR(pa_power) graphs:
pow_param_num = 10
param_num = 22
delay_num = 4
slot_num = 4

f = 10 # MHz
fs = 245.76 # MHz
nfft = 512

figure = 1
ylim = [-50, -10]

one_dim_folder = os.path.join("..", f"{1}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay")

for j_folder, folder in enumerate([one_dim_folder, two_dim_folder]):

    tx_train = np.load(os.path.join(folder, 'x.npy'))
    pa_out_train = np.load(os.path.join(folder, 'd.npy'))
    model_out_train = np.load(os.path.join(folder, 'y.npy'))
    tx_test = np.load(os.path.join(folder, 'x_test.npy'))
    pa_out_test = np.load(os.path.join(folder, 'd_test.npy'))
    model_out_test = np.load(os.path.join(folder, 'y_test.npy'))
    
    sig_len_train = int(len(tx_train) / len(power_cases_train))
    sig_len_test = int(len(tx_test) / len(power_cases_test))
    
    aclr_train = []
    aclr_test = []
    
    psd_train, psd_test, psd = [], [], []
    
    for i in range(len(power_cases_train)):
        x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
        d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
        y = model_out_train[i * sig_len_train: (i + 1) * sig_len_train]
        scale = np.max(abs(x))
        x /= scale
        d /= scale
        y /= scale
        nmse = sl.nmse(d, d - y)
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_train.append(aclr_val)
        freqs, psd = sl.get_psd(x + d - y, Fs=fs, nfft=nfft)
        psd_train.append(psd)
        print(f"Case train {power_cases_train[i]:.2f} dB: ACLR = {aclr_val} dB")
        # plt.figure(i + 1)
        # plt.title(f"Power case {power_cases_train[i]} dB")
        # pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    
    for i in range(len(power_cases_test)):
        x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
        d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        y = model_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        scale = np.max(abs(x))
        x /= scale
        d /= scale
        y /= scale
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
    
    if j_folder == 0:
        psd_train_nc, psd_test_nc, psd_nc = [], [], []
        aclr_train_no_correct, aclr_test_no_correct = [], []
        # Calculate ACLR for PA output without correction
        for i in range(len(power_cases_train)):
            x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
            d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
            scale = np.max(abs(x))
            x /= scale
            d /= scale
            aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
            aclr_train_no_correct.append(aclr_val)
            freqs, psd_val = sl.get_psd(x + d, Fs=fs, nfft=nfft)
            psd_train_nc.append(psd_val)
        
        for i in range(len(power_cases_test)):
            x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
            d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
            scale = np.max(abs(x))
            x /= scale
            d /= scale
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
        plt.plot(pa_powers, aclr_no_correct, marker='o', color='black')
        
        # Concatenate arrays to write in .txt file
        aclr_no_correct_tmp = []
        aclr_no_correct_tmp.append(pa_powers)
        aclr_no_correct_tmp.append(aclr_no_correct)
        aclr_no_correct_tmp = np.array(aclr_no_correct_tmp).T
        # Save arrays to .txt file to draw picture in latex
        try:
            folder_name = "performance"
            os.mkdir(folder_name)
        except:
            pass
        np.savetxt(os.path.join(folder_name, 'aclr_no_correct.txt'), aclr_no_correct_tmp)
    
    plt.figure(figure)
    plt.plot(pa_powers[::2], aclr_train, marker='o')
    plt.plot(pa_powers[1::2], aclr_test, marker='o')
    plt.xlabel('power, mW', fontsize=13)
    plt.ylabel('ACLR, dB', fontsize=13)
    plt.ylim(ylim)
    # plt.yticks(np.arange(10, 50, 5))
    
    # Concatenate arrays to write in .txt file
    aclr_correct_train_tmp, aclr_correct_test_tmp = [], []
    aclr_correct_train_tmp.append(pa_powers[::2])
    aclr_correct_test_tmp.append(pa_powers[1::2])
    aclr_correct_train_tmp.append(aclr_train)
    aclr_correct_test_tmp.append(aclr_test)
    aclr_correct_train_tmp = np.array(aclr_correct_train_tmp).T
    aclr_correct_test_tmp = np.array(aclr_correct_test_tmp).T
    # Save arrays to .txt file to draw picture in latex
    np.savetxt(os.path.join(folder_name, f'aclr_correct_{j_folder + 1}_dim_train.txt'), aclr_correct_train_tmp)
    np.savetxt(os.path.join(folder_name, f'aclr_correct_{j_folder + 1}_dim_test.txt'), aclr_correct_test_tmp)
    
plt.legend(['TX, not corrected',
            'TXC, 1-dim. model, train signals', 
            'TXC, 1-dim. model, test signals',
            'TXC, 2-dim. model, train signals',
            'TXC, 2-dim. model, test signals'], fontsize=13, loc='upper right')
plt.grid()

psd_max_power = deepcopy(psd[-1, :])
psd_max_power_nc = deepcopy(psd_nc[-1, :])

psd_min = -75
psd_max = -21
psd[psd < psd_min] = psd_min
# psd[psd >= -46] = -46
psd_nc[psd_nc < psd_min] = psd_min
# psd_nc[psd_nc >= -46] = -46

# Draw PSD color map
pa_range = np.arange(psd_min - 1, psd_max, 1)
levels = list(pa_range)
plt.figure(figure * 100)
F, P = np.meshgrid(freqs, pa_powers)
cs = plt.contourf(F, P, psd, levels=levels, cmap ="jet")
cbar = plt.colorbar(cs) 
plt.ylabel("PA power", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
# Draw PSD for highest power
plt.figure(figure * 100 + 1)
plt.plot(freqs, psd_max_power)
# Concatenate axis to save to .txt file
psd_max_power = np.concatenate([freqs[None, :], psd_max_power[None, :]], axis=0).T
plt.ylabel("NMSE, dB", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.yticks(np.arange(-20, -110, -10))
plt.grid()
# Save arrays to .txt file to draw picture in latex
try:
    folder_name = "psd"
    os.mkdir(folder_name)
except:
    pass
np.savetxt(os.path.join(folder_name, 'psd_max_power.txt'), psd_max_power)
np.savetxt(os.path.join(folder_name, 'freq.txt'), F)
np.savetxt(os.path.join(folder_name, 'powers.txt'), P)
np.savetxt(os.path.join(folder_name, 'PSD_corrected.txt'), psd)

levels = list(pa_range)
plt.figure(figure * 200)
F, P = np.meshgrid(freqs, pa_powers)
cs = plt.contourf(F, P, psd_nc, levels=levels, cmap ="jet")
cbar = plt.colorbar(cs) 
plt.ylabel("PA power", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
# Draw PSD for highest power
plt.figure(figure * 200 + 2)
plt.plot(freqs, psd_max_power_nc)
# Concatenate axis to save to .txt file
psd_max_power_nc = np.concatenate([freqs[None, :], psd_max_power_nc[None, :]], axis=0).T
plt.ylabel("NMSE, dB", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.yticks(np.arange(-20, -110, -10))
plt.grid()
# Save arrays to .txt file to draw picture in latex
np.savetxt(os.path.join(folder_name, 'psd_max_power_nc.txt'), psd_max_power_nc)
np.savetxt(os.path.join(folder_name, 'PSD_not_corrected.txt'), psd)

# Calculate ACLR for 2D model w.r.t. the number of parameters per |x| dimension:
param_num = list(np.arange(2, 32, 2))
aclr_val_hp_param_depend, aclr_val_lp_param_depend = [], []

two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")

for p in param_num:
    folder = os.path.join(two_dim_folder, f"{p}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
    
    tx_train = np.load(os.path.join(folder, 'x.npy'))
    pa_out_train = np.load(os.path.join(folder, 'd.npy'))
    model_out_train = np.load(os.path.join(folder, 'y.npy'))
    
    sig_len_train = int(len(tx_train) / len(power_cases_train))
    
    x_hp = tx_train[-sig_len_train:]
    d_hp = pa_out_train[-sig_len_train:]
    y_hp = model_out_train[-sig_len_train:]
    
    x_lp = tx_train[sig_len_train * 1:sig_len_train * 2]
    d_lp = pa_out_train[sig_len_train * 1:sig_len_train * 2]
    y_lp = model_out_train[sig_len_train * 1:sig_len_train * 2]

    scale_hp = np.max(abs(x_hp))
    scale_lp = np.max(abs(x_lp))
    
    x_hp /= scale_hp
    d_hp /= scale_hp
    y_hp /= scale_hp
    
    x_lp /= scale_lp
    d_lp /= scale_lp
    y_lp /= scale_lp
    aclr_val_hp = aclr_fn(x_hp + d_hp - y_hp, f=f, fs=fs, nfft=nfft)
    aclr_val_lp = aclr_fn(x_lp + d_lp - y_lp, f=f, fs=fs, nfft=nfft)
    aclr_val_hp_param_depend.append(aclr_val_hp)
    aclr_val_lp_param_depend.append(aclr_val_lp)

plt.figure(figure * 300)
plt.plot(param_num, aclr_val_hp_param_depend, color='blue', marker='o')
plt.plot(param_num, aclr_val_lp_param_depend, color='red', marker='o')
plt.yticks(np.arange(-30, -54, -2))
plt.xticks(param_num)
plt.xlabel("Parameters per |x| dimension", fontsize=13)
plt.ylabel("ACLR, dB", fontsize=13)
plt.legend(['Highest PA power: -1 dBm',
            'Lowest PA power: -16 dBm'], fontsize=13)
plt.grid()
# Concatenate axis to save to .txt file
param_num = np.array(param_num)
aclr_val_hp_param_depend = np.array(aclr_val_hp_param_depend)
aclr_val_lp_param_depend = np.array(aclr_val_lp_param_depend)
param_curve_high_power = np.concatenate([param_num[None, :], aclr_val_hp_param_depend[None, :]], axis=0).T
param_curve_low_power = np.concatenate([param_num[None, :], aclr_val_lp_param_depend[None, :]], axis=0).T
# Save arrays to .txt file to draw picture in latex
try:
    folder_name = "param_curve"
    os.mkdir(folder_name)
except:
    pass
np.savetxt(os.path.join(folder_name, 'param_curve_high_power.txt'), param_curve_high_power)
np.savetxt(os.path.join(folder_name, 'param_curve_low_power.txt'), param_curve_low_power)