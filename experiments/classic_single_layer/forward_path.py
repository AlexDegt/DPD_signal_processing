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

param_num = 10
delay_num = 12
slot_num = 4
# Determine experiment name and create its directory
# exp_name = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay"
exp_name = "10_param_4_slot_61_cases_12_delay"
# exp_name = "test"

model_eval = "train"
# model_eval = "test"

add_folder = os.path.join("one_dim_lin_scale_corr_fraq_del_7_gain_mw_m16_0dBm")
curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)
# os.mkdir(save_path)

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
if model_eval == "train":
    data_path = data_path[0::2]
elif model_eval == "test":
    data_path = data_path[1::2]
else:
    raise ValueError

# For train
pa_powers = np.load(os.path.join(folder_path, "pa_powers_round.npy"))
pa_powers = list(10 ** (np.array(pa_powers) / 10))
if model_eval == "train":
    pa_powers = pa_powers[0::2]
elif model_eval == "test":
    pa_powers = pa_powers[1::2]
else:
    raise ValueError

# Model initialization
order = [param_num, 1]
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
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = 31 * 1
# chunk_size = int(213504/chunk_num)
chunk_size = int(36846 * len(data_path) * len(train_slots_ind) // chunk_num)
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
pad_zeros = 0
trans_len = int(len(delays) // 2)
dataset = dynamic_dataset_prepare(data_path, pa_powers, dtype, device, slot_num=slot_num, delay_d=delay_d,
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
    error = (d - y)[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
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
        _, d= batch_to_tensors(batch)
        targ_pow += d[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None].abs().square().sum()
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

set_weights(model, load_weights(load_path + r'/weights_best_test_' + exp_name))
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

        # [..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None]
        # print(data[1][0, 0, :].size(), data[1][0, 0, :][..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None].size())
        y.append(model(data[0])[0, 0, :][..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
        d.append(data[1][0, 0, :][..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
        x.append(data[0][0, 0, :][..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
        # print(x[-1].size(), d[-1].size(), y[-1].size())

    y_full = torch.cat(y, dim=-1).detach().cpu().numpy()
    d_full = torch.cat(d, dim=-1).detach().cpu().numpy()
    x_full = torch.cat(x, dim=-1).detach().cpu().numpy()

    if model_eval == "train":
        np.save(load_path + r'/y.npy', y_full)
        np.save(load_path + r'/d.npy', d_full)
        np.save(load_path + r'/x.npy', x_full)
    elif model_eval == "test":
        np.save(load_path + r'/y_test.npy', y_full)
        np.save(load_path + r'/d_test.npy', d_full)
        np.save(load_path + r'/x_test.npy', x_full)
    else:
        raise ValueError