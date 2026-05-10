import sys

sys.path.append('../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train
from utils import dynamic_dataset_prepare
from scipy.io import loadmat
from model import ParallelCheby2D
from copy import deepcopy

# Simulation parameters
pow_param_num = 1
param_num = 36
delay_num = 27
slot_num = 4
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = 1 # 31 * 4
# Index of signal to take parameters from
# 0 - low power -16 dBm input, -1 - high power -1 dBm input,
ind_param = 38
# Index of signal to apply model to
ind_apply = 25

device = "cuda:4"
# device = "cpu"
seed = 964
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data. Data for different cases is concatenated together
folder_path = '../../data/single_band_dynamic'
data_path = [os.path.join(folder_path, file_name) for file_name in sorted(os.listdir(folder_path), reverse=True)]
data_path = [path for path in data_path if ".mat" in path]
data_path_apply = [data_path[ind_apply]]

# For train
pa_powers = np.load(os.path.join(folder_path, "pa_powers_round.npy"))
pa_powers_param = [pa_powers[ind_param]]
pa_powers_apply = [pa_powers[ind_apply]]
pa_powers = list(10 ** (np.array(pa_powers) / 10))

# Determine experiment name and create its directory
exp_name_param = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay_power_{pa_powers_param[0]}_dBm"
exp_name_apply = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay_power_{pa_powers_apply[0]}_dBm"

add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately")
curr_path = os.getcwd()
load_path_param = os.path.join(curr_path, add_folder, exp_name_param)
save_path_apply = os.path.join(curr_path, add_folder, exp_name_apply)

# Model initialization
order = [param_num, pow_param_num]
delays = [[j, j, j] for j in range(-delay_num, delay_num + 1)]
# Define data type
# dtype = torch.complex64
dtype = torch.complex128
slot_num = 4
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(slot_num), range(slot_num), range(slot_num)
delay_d = 0
# chunk_size = int(213504/chunk_num)
chunk_size = int(36846 * len(data_path_apply) * len(train_slots_ind) // chunk_num)
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
trans_len = int(len(delays) // 2)
pad_zeros = trans_len
dataset = dynamic_dataset_prepare(data_path_apply, list(10 ** (np.array(pa_powers_apply) / 10)), dtype, device, slot_num=slot_num, delay_d=delay_d,
                        train_slots_ind=train_slots_ind, test_slots_ind=test_slots_ind, validat_slots_ind=validat_slots_ind,
                        pad_zeros=pad_zeros, batch_size=batch_size, block_size=chunk_size)

train_dataset, validate_dataset, test_dataset = dataset

# Show sizes of batches in train dataset, size of validation and test dataset
# for i in range(len(dataset)):
#     for j, batch in enumerate(dataset[i]):
#         # if j == 0:
#         # Input batch size
#         print(batch[0].size())
#         # Target batch size
#         print(batch[1].size())
#     print(j + 1)
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1]
    return x, d

def complex_mse_loss(d, y, model):
    # d = d[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
    error = (d - y)
    return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y = batch_to_tensors(signal_batch)
    return complex_mse_loss(y, model(x), model)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
# To avoid conflicts for classification task you can write:
# def quality_criterion(loss_val):
#     return loss_val
def quality_criterion(model, dataset):
    targ_pow, loss_val = 0, 0
    for batch in dataset:
        _, d = batch_to_tensors(batch)
        targ_pow += d.abs().square().sum()
        loss_val += loss(model, batch)
    return 10.0 * torch.log10((loss_val) / (targ_pow)).item()

# def quality_criterion(model, dataset):
#     input_pow, loss_val = 0, 0
#     for batch in dataset:
#         x, _= batch_to_tensors(batch)
#         input_pow += x[..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None].abs().square().sum()
#         loss_val += loss(model, batch)
#     return 10.0 * torch.log10((loss_val) / (input_pow)).item()

def load_weights(path_name, device=device):
    return torch.load(path_name, map_location=torch.device(device))

def set_weights(model, weights):
    model.load_state_dict(weights)

def get_nested_attr(module, names):
    for i in range(len(names)):
        module = getattr(module, names[i], None)
        if module is None:
            return
    return module

model = ParallelCheby2D(order, delays, dtype, device)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")
# param_names = [name for name, p in model.named_parameters()]
# params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
# print(params)

set_weights(model, load_weights(load_path_param + r'/weights_best_test_' + exp_name_param))
# set_weights(model, load_weights(load_path + r'/weights_' + exp_name))

model.eval()
with torch.no_grad():
    # train_dataset, validate_dataset, test_dataset
    dataset = validate_dataset
    NMSE = quality_criterion(model, dataset)
    print(NMSE)
    y, d, x = [], [], []
    for j, batch in enumerate(dataset):
        data = batch_to_tensors(batch)

        y.append(model(data[0])[0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
        d.append(data[1][0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
        x.append(data[0][0, 0, trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])

    y_full = torch.cat(y, dim=-1).detach().cpu().numpy()
    d_full = torch.cat(d, dim=-1).detach().cpu().numpy()
    x_full = torch.cat(x, dim=-1).detach().cpu().numpy()

    np.save(save_path_apply + f'/y_param_{pa_powers_param[0]:.1f}dBm_apply_to_{pa_powers_apply[0]:.1f}dBm.npy', y_full)
    # np.save(save_path_apply + f'/d_param_{pa_powers_param[0]:.1f}dBm_apply_to_{pa_powers_apply[0]:.1f}dBm.npy', d_full)
    # np.save(save_path_apply + f'/x_param_{pa_powers_param[0]:.1f}dBm_apply_to_{pa_powers_apply[0]:.1f}dBm.npy', x_full)