import sys, os
sys.path.insert(0, r'../lib')
import plot_lib as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat

def fir_matrix_generate(x, M, mode='same'):
    N = np.size(x)
    U = np.zeros((N+M-1, M), dtype = 'complex128')
    for i in range(M):
        U[i:i+N, i] = x.reshape((-1,))
    cut_part = np.floor(M/2).astype(int)
    U = U[cut_part:N+M-1-cut_part, :]
    if mode == 'same':
        return U
    elif mode == 'valid':
        return U[cut_part:-cut_part, :]
    else:
        raise ValueError(f"Parameter mode must equal \'same\' or \'valid\', but \'{mode}\' is given.")

# Modified numpy.ndarray
class modndarray(np.ndarray):
    # Hermitian conjugation
    def herm(x):
        return np.conj(x).T

save_path = ""

RX_datas = np.load('RX_datas_lin_scale.npy')[::-1].view(modndarray)
# RX_datas = np.load('RX_datas_15_0_step_1.npy').view(modndarray)
# RX_datas = np.load('RX_datas.npy').view(modndarray)
# RX_datas = np.load('RX_datas_half_range.npy').view(modndarray)
x = loadmat('Tx_20MHz_Fs245p76.mat')['x'].reshape(-1)[:RX_datas.shape[1]].view(modndarray)

try:
    pa_powers = np.load('pa_powers_round.npy').view(modndarray)
    names = []
    for p in pa_powers:
        int_part = str(int(abs(p)))
        frac_part = str(int(abs(round(p - int(p), 2)) * 100))
        if int_part == '0':
            int_part = '00'
        elif int_part != '0' and len(int_part) == 1:
            int_part = '0' + int_part
        if len(frac_part) == 1:
            frac_part = '0' + frac_part
        names.append('m' + int_part + '_' + frac_part)
    names = names[::-1]
except:
    # names = ['0', '3', '6', '9', '12', '15']
    # names = ['1_5', '4_5', '7_5', '10_5', '13_5']
    names = [str(int(abs(p))) for p in range(0, -16, -1)]

case_num = RX_datas.shape[0]

TX = np.zeros_like(RX_datas)
for i, p in enumerate(reversed(pa_powers)):
    # TX[i, :] = x * 10 ** (p / 10)
    TX[i, :] = x * np.sqrt(10 ** (p / 10))
    
scale = 30000 # np.max(abs(TX[0, :]))
TX /= scale

tap_num = 7

# alpha_real = [30000, 34500, 40000, 48000, 62000, 85000]
# alpha_real = [34000, 39500, 46500, 57500, 73687]
# alpha_real = [30000, 31500, 33000, 34500, 37000, 38500, 41000, 43000, 46000, 49000, 54000, 58000, 62000, 68000, 74000, 85000]
# alpha_real = list(np.arange(30000, 30000 + 5 * 80, 80)) + \
#               list(np.arange(30400, 30400 + 5 * 90, 90)) + \
#               list(np.arange(30850, 30850 + 5 * 100, 100)) + \
#               list(np.arange(31350, 31350 + 5 * 120, 120)) + \
#               list(np.arange(31950, 31950 + 5 * 160, 160)) + \
#               list(np.arange(32750, 32750 + 5 * 180, 180)) + \
#               list(np.arange(33650, 33650 + 5 * 220, 220)) + \
#               list(np.arange(34750, 34750 + 5 * 260, 260)) + \
#               list(np.arange(36050, 36050 + 5 * 400, 400)) + \
#               list(np.arange(38050, 38050 + 5 * 580, 580)) + \
#               list(np.arange(40950, 40950 + 5 * 1100, 1100)) + \
              # [49000, 51500, 55000, 59000, 65000, 81000]

for i in range(len(names)):
    # Firstly we find real gain for x
    x_scaled = TX[i, :]
    # x_scaled = x / alpha_real[i]
    # pl.plot_psd(x_scaled, RX_datas[i, :], nfig=1, clf=False)
    # pl.plot_psd(x_scaled, nfig=1, clf=False)
    # pl.plot_psd(RX_datas[i, :], nfig=1, clf=False)
    U = fir_matrix_generate(RX_datas[i, :], tap_num).view(modndarray)
    hess = U.herm() @ U
    reg = 0 # np.max(np.abs(hess)) * 1e-3 * np.eye(tap_num)
    w = np.linalg.pinv(hess + reg) @ U.herm() @ x_scaled
    pa_out = U @ w
    # pl.plot_psd(x_scaled, pa_out, pa_out - x_scaled, nfig=1, clf=False)
    pl.plot_psd(pa_out - x_scaled, nfig=1, clf=False)
    mat = {'TX': x_scaled, 'PAout': pa_out}
    savemat(os.path.join(save_path, f'aligned_{names[i]}dB_100RB_Fs245p76.mat'), mat)