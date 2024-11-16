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
chunk_num = 8 # 31 * 18

for j_exp in range(61):
    # # Determine experiment name and create its directory
    # exp_name = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay"
    # # exp_name = "10_param_4_slot_61_cases_8_delay"
    # # exp_name = "test"

    # # add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")
    # add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately")
    
    # curr_path = os.getcwd()
    # save_path = os.path.join(curr_path, add_folder, exp_name)
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
    # data_path = data_path[0::2]

    # For train
    pa_powers = np.load(os.path.join(folder_path, "pa_powers_round.npy"))
    pa_powers = list(10 ** (np.array(pa_powers) / 10))
    # pa_powers = pa_powers[0::2]

    data_path_all = deepcopy(data_path)
    pa_powers_all = deepcopy(pa_powers)
    data_path = [data_path_all[j_exp]]
    pa_powers = [pa_powers_all[j_exp]]

    # Determine experiment name and create its directory
    exp_name = f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay_power_{10 * np.log10(pa_powers[0])}_dBm"
    # exp_name = "10_param_4_slot_61_cases_8_delay"
    # exp_name = "test"

    # add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")
    add_folder = os.path.join(f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately")
    
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, add_folder, exp_name)
    os.mkdir(save_path)

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
    # chunk_size = int(213504/chunk_num)
    chunk_size = int(36846 * len(data_path) * len(train_slots_ind) // chunk_num)
    # print(chunk_size, len(data_path), len(train_slots_ind))
    # sys.exit()
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
        error = (d - y)#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
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
            targ_pow += d.abs().square().sum()
            # targ_pow += d[..., trans_len if trans_len > 0 else None: -pad_zeros if pad_zeros > 0 else None].abs().square().sum()
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

    # sys.exit()

    # Train type shows which algorithm is used for optimization.
    # train_type='sgd_auto' # gradient-based optimizer.
    # train_type='mnm_lev_marq' # Levenberg-Marquardt on base of Mixed Newton. Work only with models with complex parameters!
    train_type='ls' # LS method: 1 Mixed-Newton step
    learning_curve, best_criterion = train(model, train_dataset, loss, quality_criterion, config_train, batch_to_tensors, validate_dataset, test_dataset, 
                                            train_type=train_type, chunk_num=chunk_num, exp_name=exp_name, save_every=1, save_path=save_path, 
                                            weight_names=weight_names, device=device)

    print(f"Best NMSE: {best_criterion} dB")