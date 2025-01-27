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
from scipy.signal import get_window, welch
from model import ParallelCheby2D
from copy import deepcopy
import yaml
import shutil
import pickle

# Load yaml file
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Simulation parameters
pow_param_num = config["pow_param_num"]
param_num = config["param_num"]
delay_num = config["delay_num"]
batch_size = config["batch_size"]
chunk_num = config["chunk_num"] # 31 * 18
# Type of data to carry out model evaluation on
model_eval_types = ["train", "test"]

# The number of signal slots in dataset
slot_num = 4

# For ACLR calculation
f = 10 # MHz
fs = 245.76 # MHz
nfft = 512

device = config["device"]
seed = 964
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data. Data for different cases is concatenated together
folder_path = config["data_path"]
data_path_whole = [os.path.join(folder_path, file_name) for file_name in sorted(os.listdir(folder_path), reverse=True)]
data_path_whole = [path for path in data_path_whole if ".mat" in path]
pa_powers_whole = np.load(os.path.join(folder_path, "pa_powers_round.npy"))
pa_powers_whole = list(10 ** (np.array(pa_powers_whole) / 10))

# Determine experiment name and create its directory
if config["trial_name"] == 'None':
    exp_name = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay"
    add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")
else:
    exp_name = config["trial_name"]
    add_folder = ''

curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)

# Determine performance curves among PA output power dynamic range
aclr_train, aclr_test = [], []
aclr_train_dpdoff, aclr_test_dpdoff = [], []
aclr_whole = np.zeros(len(pa_powers_whole),)
aclr_whole_dpdoff = np.zeros(len(pa_powers_whole),)

for model_eval in model_eval_types:
    if model_eval == "train":
        data_path = data_path_whole[0::2]
        pa_powers = pa_powers_whole[0::2]
    elif model_eval == "test":
        data_path = data_path_whole[1::2]
        pa_powers = pa_powers_whole[1::2]
    else:
        raise ValueError

    # Model initialization
    order = [param_num, pow_param_num]
    delays = [[j, j, j] for j in range(-delay_num, delay_num + 1)]
    # Define data type
    # dtype = torch.complex64
    dtype = torch.complex128
    # Indices of slots which are chosen to be included in train/test set (must be of a range type).
    # Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
    # In full-batch mode train, validation and test dataset are the same.
    # In mini-batch mode validation and test dataset are the same.
    train_slots_ind, validat_slots_ind, test_slots_ind = range(slot_num), range(slot_num), range(slot_num)
    delay_d = 0

    # Calculate whole signal length
    input = []
    for path in data_path:
        mat = loadmat(path)
        input_tensor = torch.tensor(mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        input.append(input_tensor)
    sig_len = torch.cat(input, dim=1).numel()
    slot_len = int(sig_len / len(pa_powers))

    # Calculate length of chunk to accumulate hessian and gradient in
    chunk_size = int(sig_len // chunk_num)
    # L2 regularization parameter
    alpha = 0.0
    # Configuration file
    config_train = None
    # Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
    # Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
    # NO additional padding in the input batches is required.
    trans_len = int(len(delays) // 2)
    pad_zeros = trans_len
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
        # d = d[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
        error = (d - y)
        return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

    def loss(model, signal_batch):
        x, y = batch_to_tensors(signal_batch)
        return complex_mse_loss(y, model(x), model)
    # This function is used only for telecom task.
    # Calculates NMSE on base of accumulated on every batch loss function
    @torch.no_grad()
    def nmse_fn(model, dataset):
        targ_pow, loss_val = 0, 0
        for batch in dataset:
            _, d = batch_to_tensors(batch)
            targ_pow += d.abs().square().sum()
            loss_val += loss(model, batch)
        return 10.0 * torch.log10((loss_val) / (targ_pow)).item()

    def aclr_fn_torch(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
        """
        Calculate Adjacent Channel Leakage Ratio using PyTorch.
        
        Parameters:
            sig (torch.Tensor): Complex signal (1D Tensor).
            f (float): Frequency scalar.
            fs (float, optional): Sampling frequency. Defaults to 1.0.
            nfft (int, optional): Number of FFT points. Defaults to 1024.
            window (str, optional): Window type (e.g., 'blackman'). Defaults to 'blackman'.
            nperseg (int, optional): Segment length for Welch's method. Defaults to None.
            noverlap (int, optional): Number of overlapping points. Defaults to None.
        
        Returns:
            float: ACLR in dB.
        """
        if nperseg is None:
            nperseg = nfft  # Default segment length matches FFT size
        if noverlap is None:
            noverlap = nperseg // 2  # Default overlap is 50%

        # Convert the window to a torch tensor
        win = torch.tensor(get_window(window, nperseg, fftbins=True), dtype=torch.float32).to(sig.device)

        # Reshape the signal into overlapping segments
        step = nperseg - noverlap
        num_segments = (len(sig) - noverlap) // step
        segments = torch.stack([
            sig[i * step:i * step + nperseg] for i in range(num_segments)
        ])

        # Apply the window and compute the FFT
        segments = segments * win  # Broadcasting window to each segment
        fft_segments = torch.fft.fft(segments, nfft, dim=-1)
        psd = torch.mean(torch.abs(fft_segments) ** 2, dim=0)  # Welch PSD estimate

        # FFT shift
        psd = torch.fft.fftshift(psd)
        freqs = torch.fft.fftshift(torch.fft.fftfreq(nfft, d=1/fs))
        
        # Compute the indices for the frequency range
        ind1 = (nfft // 2) - int(torch.ceil(torch.tensor(nfft * f / fs)))
        ind2 = (nfft // 2) + int(torch.ceil(torch.tensor(nfft * f / fs)))
        guard = 4  # Guard band size
        
        # Calculate ACLR
        numerator = torch.sum(psd[ind2 + guard: 2 * ind2 - ind1 + guard])
        denominator = torch.sum(psd[ind1: ind2])
        aclr = 10 * torch.log10(numerator / denominator)
        
        return aclr.item()  # Return as a scalar

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

    set_weights(model, load_weights(os.path.join(load_path, r'weights_best_test_' + exp_name)))

    model.eval()
    with torch.no_grad():
        # train_dataset, validate_dataset, test_dataset
        dataset = validate_dataset
        NMSE = nmse_fn(model, dataset)
        print(f"NMSE on whole {model_eval} dataset is {NMSE:.3f} dB")
        y, d, x = [], [], []
        for j, batch in enumerate(dataset):
            data = batch_to_tensors(batch)

            y.append(model(data[0])[0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
            d.append(data[1][0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
            x.append(data[0][0, 0, trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])

        y_full_tensor = torch.cat(y, dim=-1).to(device)
        d_full_tensor = torch.cat(d, dim=-1).to(device)
        x_full_tensor = torch.cat(x, dim=-1).to(device)
        y_full_numpy = y_full_tensor.detach().cpu().numpy()
        d_full_numpy = d_full_tensor.detach().cpu().numpy()
        x_full_numpy = x_full_tensor.detach().cpu().numpy()

        # Calculate ACLR for each of the PA output power cases
        for i_pow in range(len(pa_powers)):
            aclr_val = aclr_fn_torch((x_full_tensor + d_full_tensor - y_full_tensor)[i_pow * slot_len: (i_pow + 1) * slot_len][trans_len:-trans_len], f=f, fs=fs, nfft=nfft)
            aclr_val_dpdoff = aclr_fn_torch((x_full_tensor + d_full_tensor)[i_pow * slot_len: (i_pow + 1) * slot_len][trans_len:-trans_len], f=f, fs=fs, nfft=nfft)
            if model_eval == "train":
                aclr_train.append(aclr_val)
                aclr_train_dpdoff.append(aclr_val_dpdoff)
            elif model_eval == "test":
                aclr_test.append(aclr_val)
                aclr_test_dpdoff.append(aclr_val_dpdoff)

        if model_eval == "train":
            aclr_whole[::2] = aclr_train
            aclr_whole_dpdoff[::2] = aclr_train_dpdoff
            np.save(os.path.join(load_path, r'y.npy'), y_full_numpy)
            np.save(os.path.join(load_path, r'd.npy'), d_full_numpy)
            np.save(os.path.join(load_path, r'x.npy'), x_full_numpy)
        elif model_eval == "test":
            aclr_whole[1::2] = aclr_test
            aclr_whole_dpdoff[1::2] = aclr_test_dpdoff
            np.save(os.path.join(load_path, r'y_test.npy'), y_full_numpy)
            np.save(os.path.join(load_path, r'd_test.npy'), d_full_numpy)
            np.save(os.path.join(load_path, r'x_test.npy'), x_full_numpy)
        else:
            raise ValueError

# Determine powers for model input for train and test correspondingly
out_power_raugh = np.array([-11.6, -10.5, -9.5, -8.5, -7.6, -6.8, -6, -5.2, -4.5, -3.8, -3.1, -2.5, -1.9, -1.3, -0.8, -0.4])
in_power_ref = 10 ** (np.arange(-16, 0, 1) / 10)
out_power_ref = 10 ** ((30 + out_power_raugh) / 10)
in_power = list(10 ** ((np.load(os.path.join(folder_path, "pa_powers_round.npy")))/ 10))
# out_powers in W
pa_powers_out = np.interp(in_power, in_power_ref, out_power_ref) / 1000

with open(os.path.join(load_path, r'aclr_perform.pkl'), "wb") as file:
    pickle.dump({"ACLR": aclr_whole, "power_linear": pa_powers_out}, file)
with open(os.path.join(load_path, r'aclr_perform_dpdoff.pkl'), "wb") as file:
    pickle.dump({"ACLR": aclr_whole_dpdoff, "power_linear": pa_powers_out}, file)