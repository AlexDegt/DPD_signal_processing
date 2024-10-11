# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:36:24 2024

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

tx = np.load('x.npy')
pa_out = np.load('d.npy')
model_out = np.load('y.npy')

sig_len = int(len(tx) / 6)
tx = tx[-sig_len:]
pa_out = pa_out[-sig_len:]
model_out = model_out[-sig_len:]

def aclr(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
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

nfft = 1024
f = 10
fs = 245.76

ACLR_dpd_off = aclr(tx + pa_out, f=f, fs=fs)
ACLR_dpd_on = aclr(tx + pa_out - model_out, f=f, fs=fs)
print(f"DPD off: ACLR = {ACLR_dpd_off} dB")
print(f"DPD on: ACLR = {ACLR_dpd_on} dB")

win = signal.get_window('blackman', nfft, True)
freqs, psd = signal.welch(tx + pa_out - model_out, fs, win, return_onesided=False, detrend=False, nperseg=None, noverlap=None)
freqs = np.fft.fftshift(freqs)
psd = np.fft.fftshift(psd)
ind1 = (nfft // 2) - int(np.ceil(nfft * f/fs))
ind2 = (nfft // 2) + int(np.ceil(nfft * f/fs))
# aclr = np.linalg.norm(psd[ind1: ind2])/(np.linalg.norm(psd[:ind1]) + np.linalg.norm(psd[ind2:]))
aclr = np.sum(psd[ind1: ind2])/(np.sum(psd[:ind1]) + np.sum(psd[ind2:]))
plt.plot(10 * np.log10(psd)[ind1: ind2])
plt.plot(10 * np.log10(psd)[:ind1])
plt.plot(10 * np.log10(psd)[ind2:])
print(np.linalg.norm(psd[ind1: ind2]), np.linalg.norm(psd[:ind1]), np.linalg.norm(psd[ind2:]))