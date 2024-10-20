import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.lines import Line2D
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat
import torch

plt.rcParams["font.family"] = "Times New Roman"

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

def get_psd(sig, Fs=1.0, nfft=2048, window='blackman', nperseg=None, noverlap=None):
    """ 
        Returns Power Spectral Density of input signal sig
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    return freqs, 10*np.log10(np.fft.fftshift(psd))

def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=None, is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, 
             noverlap=None, y_shift=0, color=None, fontsize=13, xlabel='frequency', ylabel='Magnitude [dB]'):
    """ Plotting power spectral density """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()
        
    if isinstance(color, list):
        assert len(signals) == len(color) or (len(color) == 1)
        for c in color:
            assert isinstance(c, str)
        if len(color) == 1:
            color *= len(signals)
    elif isinstance(color, str) or color is None:
        color = [color] * len(signals)
    else:
        raise TypeError(f"color parameter must be either of a str or list of str type, but {type(color)} is given")
      
    ax.set_xlabel(xlabel, fontsize=fontsize)
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)

    for j_sig, iisignal in enumerate(signals):
        # freqs = np.linspace(-Fs/2, Fs/2, iisignal.size)
        # plt.plot(freqs, 10*np.log10(np.fft.fftshift(np.fft.fft(iisignal))))

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, 1, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)*Fs
        freqs += xshift
        psd = 10.0*np.log10(np.fft.fftshift(psd)) + y_shift
        ax_ptr, = ax.plot(freqs, psd, color=color[j_sig])
#        ax_ptr, = ax.plot(freqs, psd, color='tab:blue')

    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if legend is not None:
        ax.legend(legend, fontsize=fontsize)
    if is_save:
        nfig.savefig(filename)
    plt.show()
    return ax_ptr

# Determine powers for model input for train and test correspondingly
out_power_raugh = np.array([-11.6, -10.5, -9.5, -8.5, -7.6, -6.8, -6, -5.2, -4.5, -3.8, -3.1, -2.5, -1.9, -1.3, -0.8, -0.4])
in_power_ref = 10 ** (np.arange(-16, 0, 1) / 10)
out_power_ref = 10 ** ((30 + out_power_raugh) / 10)
in_power = list(10 ** ((np.load('pa_powers_round.npy') - 1)/ 10))
# out_powers in W
pa_powers = np.interp(in_power, in_power_ref, out_power_ref) / 1000

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

one_dim_folder = os.path.join("..", f"{1}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", f"{22}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
# two_dim_folder = one_dim_folder

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
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_train.append(aclr_val)
        freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
        psd_train.append(psd)
        print(f"Case train {power_cases_train[i]:.2f} W: ACLR = {aclr_val} dB")
        # plt.figure(450 + j_folder)
        # pl.plot_psd(x + d, nfig=450 + j_folder, clf=False, color='red', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, not corrected'])
        # plt.figure(450 + j_folder)
        # pl.plot_psd(x + d - y, nfig=450 + j_folder, clf=False, color='blue', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, corrected'])
        if j_folder == 0:
            plt.figure(450)
            plot_psd(x + d, nfig=450, clf=False, color='red', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, not corrected'])
            plot_psd(x + d - y, nfig=450, clf=False, color='blue', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, corrected'])
        if j_folder == 1:
            plt.figure(450)
            plot_psd(x + d - y, nfig=450, clf=False, color='green', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, corrected'])
    
    for i in range(len(power_cases_test)):
        x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
        d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        y = model_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        scale = np.max(abs(x))
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_test.append(aclr_val)
        freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
        psd_test.append(psd)
        print(f"Case test {power_cases_test[i]:.2f} W: ACLR = {aclr_val} dB")
        # plt.figure(450 + j_folder)
        # pl.plot_psd(x + d, nfig=450 + j_folder, clf=False, color='red', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, not corrected'])
        # plt.figure(450 + j_folder)
        # pl.plot_psd(x + d - y, nfig=450 + j_folder, clf=False, color='blue', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX', f'TXC, {j_folder + 1}-dim. model'])
        if j_folder == 0:
            plt.figure(450)
            plot_psd(x + d, nfig=450, clf=False, color='red', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, not corrected'])
            plot_psd(x + d - y, nfig=450, clf=False, color='blue', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX, corrected'])
        if j_folder == 1:
            plt.figure(450)
            plot_psd(x + d - y, nfig=450, clf=False, color='green', Fs=fs, xlabel='freq, MHz', ylabel='Magnitude, dB', fontsize=13, legend=['TX', 'TXC, 1-dim. model', 'TXC, 2-dim. model'])
        legend_lines = [
                Line2D([0], [0], color='red', lw=2, label='TX'),
                Line2D([0], [0], color='blue', lw=2, label='TXC, 1-dim. model'),
                Line2D([0], [0], color='green', lw=2, label='TXC, 2-dim. model'),
            ]
        plt.legend(handles=legend_lines, fontsize=13)
        plt.xticks(np.arange(-140, 140, 20))
        plt.xlim([-122.88, 122.88])
    
    
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
            freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
            psd_train_nc.append(psd_val)
        
        for i in range(len(power_cases_test)):
            x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
            d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
            scale = np.max(abs(x))
            x /= scale
            d /= scale
            aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
            aclr_test_no_correct.append(aclr_val)
            freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
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
    plt.xlabel('PA output power, W', fontsize=13)
    plt.ylabel('ACLR, dB', fontsize=13)
    plt.ylim(ylim)
    plt.yticks(np.arange(-5, -60, -5))
    
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
psd_min_power = deepcopy(psd[0, :])
psd_min_power_nc = deepcopy(psd_nc[0, :])

psd_min = -75
psd_max = -20
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
cbar.set_label("Magnitude, dB", rotation=270, labelpad=15, fontsize=13)
plt.ylabel("PA output power, W", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.xticks(np.arange(-140, 140, 20))
plt.xlim([-122.88, 122.88])
# Draw PSD for highest power
plt.figure(figure * 100 + 1)
plt.plot(freqs, psd_max_power_nc, color='red')
plt.plot(freqs, psd_max_power, color='blue')
plt.ylabel("Magnitude, dB", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.yticks(np.arange(-20, -100, -10))
plt.ylim([-90, -10])
plt.xticks(np.arange(-140, 140, 20))
plt.xlim([-122.88, 122.88])
plt.legend(['TX, 0.912 W', 'TXC, 0.912 W'], fontsize=13)
plt.grid()

levels = list(pa_range)
plt.figure(figure * 200)
F, P = np.meshgrid(freqs, pa_powers)
cs = plt.contourf(F, P, psd_nc, levels=levels, cmap ="jet")
cbar = plt.colorbar(cs)
cbar.set_label("Magnitude, dB", rotation=270, labelpad=15, fontsize=13)
plt.ylabel("PA output power, W", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.xticks(np.arange(-140, 140, 20))
plt.xlim([-122.88, 122.88])
# Draw PSD for highest power
plt.figure(figure * 200 + 2)
plt.plot(freqs, psd_min_power_nc, color='red')
plt.plot(freqs, psd_min_power, color='blue')
plt.ylabel("Magnitude, dB", fontsize=13)
plt.xlabel("freq, MHz", fontsize=13)
plt.yticks(np.arange(-20, -100, -10))
plt.ylim([-90, -10])
plt.xticks(np.arange(-140, 140, 20))
plt.xlim([-122.88, 122.88])
plt.legend(['TX, 0.069 W', 'TXC, 0.069 W'], fontsize=13)
plt.grid()

# Concatenate axis to save to .txt file
psd_max_power_tmp = np.concatenate([freqs[None, :], psd_max_power[None, :]], axis=0).T
psd_max_power_nc_tmp = np.concatenate([freqs[None, :], psd_max_power_nc[None, :]], axis=0).T
psd_min_power_tmp = np.concatenate([freqs[None, :], psd_min_power[None, :]], axis=0).T
psd_min_power_nc_tmp = np.concatenate([freqs[None, :], psd_min_power_nc[None, :]], axis=0).T
# Save arrays to .txt file to draw picture in latex
try:
    folder_name = "psd"
    os.mkdir(folder_name)
except:
    pass
np.savetxt(os.path.join(folder_name, 'psd_max_power.txt'), psd_max_power_tmp)
np.savetxt(os.path.join(folder_name, 'psd_max_power_nc.txt'), psd_max_power_nc_tmp)
np.savetxt(os.path.join(folder_name, 'psd_min_power.txt'), psd_min_power_tmp)
np.savetxt(os.path.join(folder_name, 'psd_min_power_nc.txt'), psd_min_power_nc_tmp)
np.savetxt(os.path.join(folder_name, 'freq.txt'), F)
np.savetxt(os.path.join(folder_name, 'powers.txt'), P)
np.savetxt(os.path.join(folder_name, 'PSD_corrected.txt'), psd)
np.savetxt(os.path.join(folder_name, 'PSD_not_corrected.txt'), psd_nc)

# Calculate ACLR for 2D model w.r.t. the number of parameters per |x| dimension:
param_num = list(np.arange(2, 32, 2))
aclr_val_param_depend_train, aclr_val_param_depend_test = [], []

two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")

for p in param_num:
    folder = os.path.join(two_dim_folder, f"{p}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
    
    tx_train = np.load(os.path.join(folder, 'x.npy'))
    pa_out_train = np.load(os.path.join(folder, 'd.npy'))
    model_out_train = np.load(os.path.join(folder, 'y.npy'))
    
    tx_test = np.load(os.path.join(folder, 'x_test.npy'))
    pa_out_test = np.load(os.path.join(folder, 'd_test.npy'))
    model_out_test = np.load(os.path.join(folder, 'y_test.npy'))
    
    sig_len_train = int(len(tx_train) / len(power_cases_train))
    sig_len_test = int(len(tx_test) / len(power_cases_test))
    
    aclr_val_curr_param_depend = []
    for j_case in range(len(power_cases_train)):
        x = tx_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
        d = pa_out_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
        y = model_out_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
    
        scale = np.max(abs(x))
        
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_val_curr_param_depend.append(aclr_val)
    aclr_val_param_depend_train.append(aclr_val_curr_param_depend)
    
    aclr_val_curr_param_depend = []
    for j_case in range(len(power_cases_test)):
        x = tx_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
        d = pa_out_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
        y = model_out_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
    
        scale = np.max(abs(x))
        
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_val_curr_param_depend.append(aclr_val)
    aclr_val_param_depend_test.append(aclr_val_curr_param_depend)
    
aclr_val_param_depend_train = np.array(aclr_val_param_depend_train).T
aclr_val_param_depend_test = np.array(aclr_val_param_depend_test).T

aclr_val_param_depend = np.zeros((len(pa_powers), len(param_num)))
# aclr_val_param_depend = np.concatenate([aclr_val_param_depend_train, aclr_val_param_depend_test], axis=0)
aclr_val_param_depend[::2] = aclr_val_param_depend_train
aclr_val_param_depend[1::2] = aclr_val_param_depend_test

plt.figure(figure * 300)
for j_param, p in enumerate(pa_powers):
    plt.plot(param_num, aclr_val_param_depend[j_param, :], marker='o')
# plt.plot(param_num, aclr_val_param_depend[-1, :], color='blue', marker='o')
# plt.plot(param_num, aclr_val_param_depend[1, :], color='red', marker='o')
plt.yticks(np.arange(-30, -54, -2))
plt.xticks(param_num)
plt.xlabel("Parameters per |x| dimension", fontsize=13)
plt.ylabel("ACLR, dB", fontsize=13)
# plt.legend(['PA output power: 0.912 W',
#             'PA output power: 0.069 W'], fontsize=13)
plt.grid()
# Concatenate axis to save to .txt file
# param_num = np.array(param_num)
# aclr_val_hp_param_depend = np.array(aclr_val_param_depend[-1, :])
# aclr_val_lp_param_depend = np.array(aclr_val_param_depend[0, :])
# param_curve_high_power = np.concatenate([param_num[None, :], aclr_val_hp_param_depend[None, :]], axis=0).T
# param_curve_low_power = np.concatenate([param_num[None, :], aclr_val_lp_param_depend[None, :]], axis=0).T
# # Save arrays to .txt file to draw picture in latex
# try:
#     folder_name = "param_curve"
#     os.mkdir(folder_name)
# except:
#     pass
# np.savetxt(os.path.join(folder_name, 'param_curve_high_power.txt'), param_curve_high_power)
# np.savetxt(os.path.join(folder_name, 'param_curve_low_power.txt'), param_curve_low_power)