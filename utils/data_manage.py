import torch
from typing import Tuple, Union, Iterable, List, Optional
import torch.nn.functional as F
import scipy.signal as signal
from scipy.io import loadmat
import sys
import numpy as np
from scipy.io import loadmat

OptionalTensor = Union[torch.Tensor, None]
OptionalStr = Union[str, None]
OptionalInt = Union[int, None]
DatasetType = Union[Tuple[Iterable, ...], Tuple[Tuple[Iterable, ...], ...]]
ListOfStr = List[str]
ListOfFloat = List[float]

class ResampleDataset(torch.utils.data.Dataset):
    """
    The dataset class that extracts batches in a tuple type, where the first
    tuple element contains input batch part, the second tuple element contains 
    target batch part.
    """
    def __init__(self, data: Tuple[torch.Tensor], batch_size: OptionalInt = None, downsample_ratio: OptionalInt = None,
                 aggregate_power: str = "concat"):
        super(ResampleDataset, self).__init__()
        if downsample_ratio is None:
            downsample_ratio = 1
        if batch_size is None:
            batch_size = 1
            self.batch_num = 1
        else:
            self.batch_num = int(np.ceil(data[0].shape[0]/batch_size))
        self.data = tuple((data[0], data[1]))
        self.batch_size = int(batch_size)
        self.aggregate_power = aggregate_power
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        if self.aggregate_power == "concat":
            if index < self.batch_num -  1:
                return tuple((self.data[0][index*self.batch_size:(index+1)*self.batch_size, ...], 
                                self.data[1][index*self.batch_size:(index+1)*self.batch_size, ...]))
            if index == self.batch_num -  1:
                return tuple((self.data[0][index*self.batch_size:, ...], 
                                self.data[1][index*self.batch_size:, ...]))
        elif self.aggregate_power == "batch":
            if index < self.batch_num -  1:
                return tuple((self.data[0][index*self.batch_size:(index+1)*self.batch_size, ...][0, ...], 
                                self.data[1][index*self.batch_size:(index+1)*self.batch_size, ...][0, ...]))
            if index == self.batch_num -  1:
                return tuple((self.data[0][index*self.batch_size:, ...][0, ...], 
                                self.data[1][index*self.batch_size:, ...][0, ...]))
        else:
            raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {self.aggregate_power} is given.")
    def __len__(self) -> int:
        return self.batch_num

def dynamic_dataset_prepare(data_path: ListOfStr, pa_powers: ListOfFloat, dtype: torch.dtype = torch.complex128, device: str = 'cuda', batch_size: OptionalInt = None, 
                    block_size: OptionalInt = None, slot_num: OptionalInt = None, pad_zeros: OptionalInt = None, 
                    delay_d: OptionalInt = None, train_slots_ind: range = range(1), validat_slots_ind: range = range(1),
                    test_slots_ind: range = range(1), aggregate_power: str = "concat", return_scales=False) -> DatasetType:
    """
    The method extracts input and target data for the mat file, normalizes and resamples if necessary.
    Then it divides input and target tensors into the batches and loads them into the dataloader.

    Args:
        data_path (list of str): List of pathes to mat-files with dynamic data. Each mat-file contains:
            input (numpy.ndarray): 1d array with shape (1, arr.shape), which contains input data samples.
            target (numpy.ndarray): 1d array with shape (1, arr.shape), which contains target data samples.
            noise_floor (numpy.ndarray): 1d array with shape (1, arr.shape), which contains noise floor samples.
        pa_powers (list of float): List PA output powers, which correspond to data within data_path mat-files.
        dtype (torch.dtype): The type of tensor to convert content of the mat-file to. Defaults is torch.complex128.
        device (str): The device to load dataset to. Defaults is 'cpu'.
        slot_num (int, optional): The number of slots to divide the whole dataset into.
        pad_zeros (int, optional): The number of zeros to add to the beginning and to the end of the signal.
        delay_d (int, optional): The value of desired and noise floor signals shift. delay_d > 0 corresponds to
            samples shift left, delay_d < 0 corresponds to samples shift right. If delay_d equal zero or None, 
            then there is no shift. Defaults is "None".
        train_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
        validat_slots_ind (range): Used only for hold-out cross-validation. Indices of the slots which are chosen for validation dataset. 
            A range with step 1. Defaults is range(1).
        test_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
        aggregate_power (str): The flag which illustrates how to aggragate data corresponding to different PA output powers.
            "concat" - concatenate signals into single vector,
            "batch" - put signals for different powers into batch dimension.

    Returns:
        Tuple of iterables.
    """
    
    if pad_zeros is None:
        pad_zeros = 0

    if batch_size is None:
        batch_size = 1

    assert len(data_path) == len(pa_powers), "Number of dynamic cases in data_path must equal number of corresponding PA output powers."
    dynam_case_num = len(data_path)

    input, target = [], []
    scales = []
    scales_target = []
    for path in data_path:
        mat = loadmat(path)
        input_tensor = torch.tensor(mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        target_tensor = torch.tensor(mat['PAout'][0, :] - mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        
        # Signals standartization
        scales.append(input_tensor.abs().max().item())
        scales_target.append(target_tensor.abs().max().item())
        # input_tensor /= scales[-1]
        # target_tensor /= scales_target[-1]
        # target_tensor /= target_tensor.abs().max()
        
        # input.append(input_tensor / input_tensor.abs().max().item())
        input.append(input_tensor)
        # target.append(target_tensor / input_tensor.abs().max().item())
        target.append(target_tensor)
        

    pa_list = [pa_pow * torch.ones(1, 1, input[0].numel()) for pa_pow in pa_powers]
    pa_powers = torch.cat(pa_list, dim=1).to(device).to(dtype)

    # Additional features standartization
    pa_powers -= pa_powers.abs().min()
    pa_powers /= pa_powers.abs().max()

    input = torch.cat(input, dim=1)
    target = torch.cat(target, dim=1)

    # Scale whole input signal to some interval
    input_scale = input.abs().max()
    if input_scale > 0:
        input = input / input_scale
        target = target / input_scale

    input = torch.cat([input[:, None, ...], pa_powers[:, None, ...]], dim=1)

    if delay_d is not None and delay_d != 0:
        target = torch.roll(target, -delay_d, dims=-1)

    input = input / 1
    target = target / 1

    assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
        "All train slots indices (argument train_slots_ind) must be positive and lower, than number of slots (argument slot_num)."
    assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
        "All validation slots indices (argument validat_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
        "All test slots indices (argument test_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert type(train_slots_ind) == range and type(test_slots_ind) == range and type(validat_slots_ind), \
        f"Types of train, validation and test indices must be a range, but {type(train_slots_ind)}, {type(validat_slots_ind)} and {type(test_slots_ind)} are given correspondingly."
    assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
        f"Step of indices ranges train_slots_ind, validat_slots_ind and test_slots_ind must equal 1, but {train_slots_ind.step}, {validat_slots_ind.step} and {test_slots_ind.step} are given correspondingly."

    slot_input_size = int(input.shape[-1]/slot_num)
    slot_target_size = int(target.shape[-1]/slot_num)
    
    input_train_size = slot_input_size * len(train_slots_ind)
    target_train_size = slot_target_size * len(train_slots_ind)
    input_validat_size = slot_input_size * len(validat_slots_ind)
    target_validat_size = slot_target_size * len(validat_slots_ind)
    input_test_size = slot_input_size * len(test_slots_ind)
    target_test_size = slot_target_size * len(test_slots_ind)
    
    if block_size is None: 
        block_size = input_train_size
    block_size_target = block_size
    
    if block_size is None: 
        block_size = input_test_size
    block_size_test = block_size
    block_size_test_target = block_size_test

    if block_size is None: 
        block_size = input_validat_size
    block_size_validat = block_size
    block_size_validat_target = block_size_validat
    
    dataset = list()
    
    train_input_set = input[..., train_slots_ind[0] * slot_input_size: train_slots_ind[0] * slot_input_size + input_train_size]
    train_target_set = target[..., train_slots_ind[0] * slot_target_size: train_slots_ind[0] * slot_target_size + target_train_size]
    if aggregate_power == "concat":
        train_input_set = train_input_set.reshape(1, 2, -1)
        train_target_set = train_target_set.reshape(1, 1, -1)
    elif aggregate_power == "batch":
        train_input_set = train_input_set[0, ...].permute(1, 0, 2)
        train_target_set = train_target_set.permute(1, 0, 2)
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    train_input_set = F.pad(train_input_set, (pad_zeros, pad_zeros))
    
    # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
    step = int(block_size)
    pad_unfold_input = (step + 2*pad_zeros - train_input_set.size()[-1] % step) % step
    train_input_set = F.pad(train_input_set, (0, pad_unfold_input))
    step = int(block_size_target)
    pad_unfold_target = (step - train_target_set.size()[-1] % step) % step
    train_target_set = F.pad(train_target_set, (0, pad_unfold_target))

    if aggregate_power == "concat":
        train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size))[0, ...].permute(1, 0, 2)
        train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target))[0, ...].permute(1, 0, 2)
    elif aggregate_power == "batch":
        train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size)).permute(2, 0, 1, 3)
        train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target)).permute(2, 0, 1, 3)
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    train_set = tuple((train_input_set, train_target_set))
    train_set = ResampleDataset(train_set, batch_size=batch_size, aggregate_power=aggregate_power)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=None)
    
    validat_input_set = input[..., validat_slots_ind[0] * slot_input_size: validat_slots_ind[0] * slot_input_size + input_validat_size]
    validat_target_set = target[..., validat_slots_ind[0] * slot_target_size: validat_slots_ind[0] * slot_target_size + target_validat_size]
    if aggregate_power == "concat":
        validat_input_set = validat_input_set.reshape(1, 2, -1)
        validat_target_set = validat_target_set.reshape(1, 1, -1)
    elif aggregate_power == "batch":
        validat_input_set = validat_input_set[0, ...].permute(1, 0, 2)
        validat_target_set = validat_target_set.permute(1, 0, 2)
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    validat_input_set = F.pad(validat_input_set, (pad_zeros, pad_zeros))

    # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
    step = int(block_size_validat)
    pad_unfold_input = (step + 2*pad_zeros - validat_input_set.size()[-1] % step) % step
    validat_input_set = F.pad(validat_input_set, (0, pad_unfold_input))
    step = int(block_size_validat_target)
    pad_unfold_target = (step - validat_target_set.size()[-1] % step) % step
    validat_target_set = F.pad(validat_target_set, (0, pad_unfold_target))

    if aggregate_power == "concat":
        validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat)[0, ...].permute(1, 0, 2)
        validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target)[0, ...].permute(1, 0, 2)
    elif aggregate_power == "batch":
        validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat).permute(2, 0, 1, 3)
        validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target).permute(2, 0, 1, 3)
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    validat_set = tuple((validat_input_set, validat_target_set))
    validat_set = ResampleDataset(validat_set, batch_size=batch_size, aggregate_power=aggregate_power)
    validat_set = torch.utils.data.DataLoader(validat_set, batch_size=None)

    test_input_set = input[..., test_slots_ind[0] * slot_input_size: test_slots_ind[0] * slot_input_size + input_test_size]
    test_target_set = target[..., test_slots_ind[0] * slot_target_size: test_slots_ind[0] * slot_target_size + target_test_size]
    if aggregate_power == "concat":
        test_input_set = test_input_set.reshape(1, 2, -1)
        test_target_set = test_target_set.reshape(1, 1, -1)
    elif aggregate_power == "batch":
        test_input_set = test_input_set[0, ...].permute(1, 0, 2)
        test_target_set = test_target_set.permute(1, 0, 2)
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    test_input_set = F.pad(test_input_set, (pad_zeros, pad_zeros))

    # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
    step = int(block_size_test)
    pad_unfold_input = (step + 2*pad_zeros - test_input_set.size()[-1] % step) % step
    test_input_set = F.pad(test_input_set, (0, pad_unfold_input))
    step = int(block_size_test_target)
    pad_unfold_target = (step - test_target_set.size()[-1] % step) % step
    test_target_set = F.pad(test_target_set, (0, pad_unfold_target))

    # # Used for MNM for memoty economy while test data performance calculation
    # if aggregate_power == "concat":
    #     test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
    #     test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
    # elif aggregate_power == "batch":
    #     test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test).permute(2, 0, 1, 3)
    #     test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target).permute(2, 0, 1, 3)
    # else:
    #     raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    
    # Used only for SGD for faster test data performance calculation
    if aggregate_power == "concat":
        test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
        test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
    elif aggregate_power == "batch":
        test_input_set = test_input_set[:, None, ...]
        test_target_set = test_target_set[:, None, ...]
    else:
        raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")

    test_set = tuple((test_input_set, test_target_set))
    test_set = ResampleDataset(test_set, batch_size=batch_size, aggregate_power=aggregate_power)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=None)
    
    dataset.append(tuple((train_set, validat_set, test_set)))

    if return_scales:
        return dataset[0], [scales, scales_target]
    else:
        return dataset[0]

# def dynamic_permute_dataset_prepare(data_path: ListOfStr, pa_powers: ListOfFloat, dtype: torch.dtype = torch.complex128, device: str = 'cuda', batch_size: OptionalInt = None, 
#                     block_size: OptionalInt = None, slot_num: OptionalInt = None, pad_zeros: OptionalInt = None, 
#                     delay_d: OptionalInt = None, train_slots_ind: range = range(1), validat_slots_ind: range = range(1),
#                     test_slots_ind: range = range(1), aggregate_power: str = "concat", return_scales=False,
#                     signal_ind = None) -> DatasetType:
#     """
#     The method extracts input and target data for the mat file, normalizes and resamples if necessary.
#     Then it divides input and target tensors into the batches and loads them into the dataloader.

#     Args:
#         data_path (list of str): List of pathes to mat-files with dynamic data. Each mat-file contains:
#             input (numpy.ndarray): 1d array with shape (1, arr.shape), which contains input data samples.
#             target (numpy.ndarray): 1d array with shape (1, arr.shape), which contains target data samples.
#             noise_floor (numpy.ndarray): 1d array with shape (1, arr.shape), which contains noise floor samples.
#         pa_powers (list of float): List PA output powers, which correspond to data within data_path mat-files.
#         dtype (torch.dtype): The type of tensor to convert content of the mat-file to. Defaults is torch.complex128.
#         device (str): The device to load dataset to. Defaults is 'cpu'.
#         slot_num (int, optional): The number of slots to divide the whole dataset into.
#         pad_zeros (int, optional): The number of zeros to add to the beginning and to the end of the signal.
#         delay_d (int, optional): The value of desired and noise floor signals shift. delay_d > 0 corresponds to
#             samples shift left, delay_d < 0 corresponds to samples shift right. If delay_d equal zero or None, 
#             then there is no shift. Defaults is "None".
#         train_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
#         validat_slots_ind (range): Used only for hold-out cross-validation. Indices of the slots which are chosen for validation dataset. 
#             A range with step 1. Defaults is range(1).
#         test_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
#         aggregate_power (str): The flag which illustrates how to aggragate data corresponding to different PA output powers.
#             "concat" - concatenate signals into single vector,
#             "batch" - put signals for different powers into batch dimension.

#     Returns:
#         Tuple of iterables.
#     """
    
#     if pad_zeros is None:
#         pad_zeros = 0

#     if batch_size is None:
#         batch_size = 1

#     assert len(data_path) == len(pa_powers), "Number of dynamic cases in data_path must equal number of corresponding PA output powers."
#     dynam_case_num = len(data_path)

#     input = []
#     for path in data_path:
#         mat = loadmat(path)
#         input_tensor = torch.tensor(mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
#         input.append(input_tensor)
#     input = torch.cat(input, dim=1)
#     input_scale = input.abs().max()
#     pa_powers_min = pa_powers.abs().min()
#     pa_powers_max = pa_powers.abs().max()

#     if signal_ind is None:
#         signal_ind = np.random.randint(0, len(data_path))
#     else:
#         signal_ind = np.array(signal_ind)

#     data_path = np.array(data_path)[signal_ind].tolist()
#     pa_powers = np.array(pa_powers)[signal_ind].tolist()

#     signal_num = len(signal_ind)

#     input, target = [], []
#     scales = []
#     scales_target = []
#     for path in data_path:
#         mat = loadmat(path)
#         input_tensor = torch.tensor(mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
#         target_tensor = torch.tensor(mat['PAout'][0, :] - mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        
#         # Signals standartization
#         scales.append(input_tensor.abs().max().item())
#         scales_target.append(target_tensor.abs().max().item())
#         # input_tensor /= scales[-1]
#         # target_tensor /= scales_target[-1]
#         # target_tensor /= target_tensor.abs().max()
        
#         input.append(input_tensor)
#         target.append(target_tensor)

#     pa_list = [pa_pow * torch.ones(1, 1, input[0].numel()) for pa_pow in pa_powers]
#     pa_powers = torch.cat(pa_list, dim=1).to(device).to(dtype)

#     # Additional features standartization
#     pa_powers -= pa_powers_min
#     pa_powers /= pa_powers_max

#     input = torch.cat(input, dim=1)
#     target = torch.cat(target, dim=1)

#     # Scale whole input signal to some interval
#     input /= input_scale
#     # target /= target.abs().max()

#     input = torch.cat([input[:, None, ...], pa_powers[:, None, ...]], dim=1)

#     if delay_d is not None and delay_d != 0:
#         target = torch.roll(target, -delay_d, dims=-1)

#     input = input / 1
#     target = target / 1

#     assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
#         "All train slots indices (argument train_slots_ind) must be positive and lower, than number of slots (argument slot_num)."
#     assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
#         "All validation slots indices (argument validat_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
#     assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
#         "All test slots indices (argument test_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
#     assert type(train_slots_ind) == range and type(test_slots_ind) == range and type(validat_slots_ind), \
#         f"Types of train, validation and test indices must be a range, but {type(train_slots_ind)}, {type(validat_slots_ind)} and {type(test_slots_ind)} are given correspondingly."
#     assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
#         f"Step of indices ranges train_slots_ind, validat_slots_ind and test_slots_ind must equal 1, but {train_slots_ind.step}, {validat_slots_ind.step} and {test_slots_ind.step} are given correspondingly."

#     slot_input_size = int(input.shape[-1]/slot_num)
#     slot_target_size = int(target.shape[-1]/slot_num)
    
#     input_train_size = slot_input_size * len(train_slots_ind)
#     target_train_size = slot_target_size * len(train_slots_ind)
#     input_validat_size = slot_input_size * len(validat_slots_ind)
#     target_validat_size = slot_target_size * len(validat_slots_ind)
#     input_test_size = slot_input_size * len(test_slots_ind)
#     target_test_size = slot_target_size * len(test_slots_ind)
    
#     if block_size is None: 
#         block_size = input_train_size
#     block_size_target = block_size
    
#     if block_size is None: 
#         block_size = input_test_size
#     block_size_test = block_size
#     block_size_test_target = block_size_test

#     if block_size is None: 
#         block_size = input_validat_size
#     block_size_validat = block_size
#     block_size_validat_target = block_size_validat
    
#     dataset = list()
    
#     train_input_set = input[..., train_slots_ind[0] * slot_input_size: train_slots_ind[0] * slot_input_size + input_train_size]
#     train_target_set = target[..., train_slots_ind[0] * slot_target_size: train_slots_ind[0] * slot_target_size + target_train_size]
#     if aggregate_power == "concat":
#         train_input_set = train_input_set.reshape(1, 2, -1)
#         train_target_set = train_target_set.reshape(1, 1, -1)
#     elif aggregate_power == "batch":
#         train_input_set = train_input_set[0, ...].permute(1, 0, 2)
#         train_target_set = train_target_set.permute(1, 0, 2)
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
#     train_input_set = F.pad(train_input_set, (pad_zeros, pad_zeros))
    
#     # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
#     step = int(block_size)
#     pad_unfold_input = (step + 2*pad_zeros - train_input_set.size()[-1] % step) % step
#     train_input_set = F.pad(train_input_set, (0, pad_unfold_input))
#     step = int(block_size_target)
#     pad_unfold_target = (step - train_target_set.size()[-1] % step) % step
#     train_target_set = F.pad(train_target_set, (0, pad_unfold_target))

#     if aggregate_power == "concat":
#         train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size))[0, ...].permute(1, 0, 2)
#         train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target))[0, ...].permute(1, 0, 2)
#     elif aggregate_power == "batch":
#         train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size)).permute(2, 0, 1, 3)
#         train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target)).permute(2, 0, 1, 3)
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
#     train_set = tuple((train_input_set, train_target_set))
#     train_set = ResampleDataset(train_set, batch_size=batch_size, aggregate_power=aggregate_power)
#     train_set = torch.utils.data.DataLoader(train_set, batch_size=None)
    
#     validat_input_set = input[..., validat_slots_ind[0] * slot_input_size: validat_slots_ind[0] * slot_input_size + input_validat_size]
#     validat_target_set = target[..., validat_slots_ind[0] * slot_target_size: validat_slots_ind[0] * slot_target_size + target_validat_size]
#     if aggregate_power == "concat":
#         validat_input_set = validat_input_set.reshape(1, 2, -1)
#         validat_target_set = validat_target_set.reshape(1, 1, -1)
#     elif aggregate_power == "batch":
#         validat_input_set = validat_input_set[0, ...].permute(1, 0, 2)
#         validat_target_set = validat_target_set.permute(1, 0, 2)
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
#     validat_input_set = F.pad(validat_input_set, (pad_zeros, pad_zeros))

#     # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
#     step = int(block_size_validat)
#     pad_unfold_input = (step + 2*pad_zeros - validat_input_set.size()[-1] % step) % step
#     validat_input_set = F.pad(validat_input_set, (0, pad_unfold_input))
#     step = int(block_size_validat_target)
#     pad_unfold_target = (step - validat_target_set.size()[-1] % step) % step
#     validat_target_set = F.pad(validat_target_set, (0, pad_unfold_target))

#     if aggregate_power == "concat":
#         validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat)[0, ...].permute(1, 0, 2)
#         validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target)[0, ...].permute(1, 0, 2)
#     elif aggregate_power == "batch":
#         validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat).permute(2, 0, 1, 3)
#         validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target).permute(2, 0, 1, 3)
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
#     validat_set = tuple((validat_input_set, validat_target_set))
#     validat_set = ResampleDataset(validat_set, batch_size=batch_size, aggregate_power=aggregate_power)
#     validat_set = torch.utils.data.DataLoader(validat_set, batch_size=None)

#     test_input_set = input[..., test_slots_ind[0] * slot_input_size: test_slots_ind[0] * slot_input_size + input_test_size]
#     test_target_set = target[..., test_slots_ind[0] * slot_target_size: test_slots_ind[0] * slot_target_size + target_test_size]
#     if aggregate_power == "concat":
#         test_input_set = test_input_set.reshape(1, 2, -1)
#         test_target_set = test_target_set.reshape(1, 1, -1)
#     elif aggregate_power == "batch":
#         test_input_set = test_input_set[0, ...].permute(1, 0, 2)
#         test_target_set = test_target_set.permute(1, 0, 2)
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
#     test_input_set = F.pad(test_input_set, (pad_zeros, pad_zeros))

#     # Pad array of signal with zeros in order not to lose data by implementation of torch.unfold
#     step = int(block_size_test)
#     pad_unfold_input = (step + 2*pad_zeros - test_input_set.size()[-1] % step) % step
#     test_input_set = F.pad(test_input_set, (0, pad_unfold_input))
#     step = int(block_size_test_target)
#     pad_unfold_target = (step - test_target_set.size()[-1] % step) % step
#     test_target_set = F.pad(test_target_set, (0, pad_unfold_target))

#     # # Used for MNM for memoty economy while test data performance calculation
#     # if aggregate_power == "concat":
#     #     test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
#     #     test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
#     # elif aggregate_power == "batch":
#     #     test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test).permute(2, 0, 1, 3)
#     #     test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target).permute(2, 0, 1, 3)
#     # else:
#     #     raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")
    
#     # Used only for SGD for faster test data performance calculation
#     if aggregate_power == "concat":
#         test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
#         test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
#     elif aggregate_power == "batch":
#         test_input_set = test_input_set[:, None, ...]
#         test_target_set = test_target_set[:, None, ...]
#     else:
#         raise ValueError(f"aggregate_power must equal \'concat\' or \'batch\', but {aggregate_power} is given.")

#     test_set = tuple((test_input_set, test_target_set))
#     test_set = ResampleDataset(test_set, batch_size=batch_size, aggregate_power=aggregate_power)
#     test_set = torch.utils.data.DataLoader(test_set, batch_size=None)
    
#     dataset.append(tuple((train_set, validat_set, test_set)))

#     if return_scales:
#         return dataset[0], [scales, scales_target]
#     else:
#         return dataset[0]

def dynamic_permute_dataset_prepare(
    data_path: ListOfStr,
    pa_powers: ListOfFloat,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cuda',
    batch_size: OptionalInt = None,
    block_size: OptionalInt = None,
    slot_num: OptionalInt = None,
    pad_zeros: OptionalInt = None,
    delay_d: OptionalInt = None,
    train_slots_ind: range = range(1),
    validat_slots_ind: range = range(1),
    test_slots_ind: range = range(1),
    aggregate_power: str = "concat",   # оставил аргумент для совместимости
    return_scales: bool = False,
    signal_ind=None
) -> DatasetType:
    """
    Логика:
    - выбираются K сигналов по signal_ind
    - train: выбранные K сигналов конкатенируются по времени и режутся на блоки длины block_size
    - valid/test: остаются K отдельными сигналами (по одному на мощность), без конкатенации между собой
    """

    if pad_zeros is None:
        pad_zeros = 0

    if batch_size is None:
        batch_size = 1

    assert slot_num is not None and slot_num > 0, "slot_num must be a positive integer."
    assert len(data_path) == len(pa_powers), (
        "Number of dynamic cases in data_path must equal number of corresponding PA output powers."
    )

    # -----------------------------
    # Проверка диапазонов слотов
    # -----------------------------
    assert isinstance(train_slots_ind, range) and isinstance(validat_slots_ind, range) and isinstance(test_slots_ind, range), \
        f"Types of train, validation and test indices must be range, but got {type(train_slots_ind)}, {type(validat_slots_ind)}, {type(test_slots_ind)}."

    assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
        f"Steps must be 1, but got {train_slots_ind.step}, {validat_slots_ind.step}, {test_slots_ind.step}."

    assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
        "All train slots indices must be >= 0 and < slot_num."
    assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
        "All validation slots indices must be >= 0 and < slot_num."
    assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
        "All test slots indices must be >= 0 and < slot_num."

    # -----------------------------
    # Выбор сигналов
    # -----------------------------
    if signal_ind is None:
        signal_ind = np.array([np.random.randint(0, len(data_path))], dtype=int)
    else:
        signal_ind = np.atleast_1d(np.array(signal_ind, dtype=int))

    data_path_sel = np.array(data_path, dtype=object)[signal_ind].tolist()
    pa_powers_sel = np.array(pa_powers)[signal_ind]
    signal_num = len(signal_ind)

    # data_path_all = np.array(data_path, dtype=object).tolist()
    # scale_inp_max = 0
    # scale_target_max = 0
    # for path in data_path_all:
    #     mat = loadmat(path)

    #     tx = torch.tensor(mat['TX'][0, :], dtype=dtype, device=device).view(1, 1, -1)#[:, :, 1104:-1104]
    #     paout = torch.tensor(mat['PAout'][0, :], dtype=dtype, device=device).view(1, 1, -1)#[:, :, 1104:-1104]
    #     target = (paout - tx)

    #     if scale_inp_max < tx.abs().max().item():
    #         scale_inp_max = tx.abs().max().item()
    #     if scale_target_max < target.abs().max().item():
    #         scale_target_max = target.abs().max().item()

    # -----------------------------
    # Загрузка выбранных K сигналов
    # input_list  -> список [1,1,N]
    # target_list -> список [1,1,N]
    # -----------------------------
    input_list = []
    target_list = []
    scales = []
    scales_target = []

    for path in data_path_sel:
        mat = loadmat(path)

        tx = torch.tensor(mat['TX'][0, :], dtype=dtype, device=device).view(1, 1, -1)#[:, :, 1104:-1104]
        paout = torch.tensor(mat['PAout'][0, :], dtype=dtype, device=device).view(1, 1, -1)#[:, :, 1104:-1104]
        target = (paout - tx)

        scales.append(tx.abs().max().item())
        scales_target.append(target.abs().max().item())

        # ACLR сильно зависит от нормировки! Не забыть обдумать!
        # input_list.append(tx / 30000) #/ tx.abs().max().item())
        input_list.append(tx) #/ tx.abs().max().item())
        # input_list.append(tx / tx.abs().max().item())
        # target_list.append(target / 30000) #/ target.abs().max().item())
        target_list.append(target)

    # [1, K, N]
    input_tensor = torch.cat(input_list, dim=1)
    target_tensor = torch.cat(target_list, dim=1)

    # Масштабируем вход по глобальному максимуму среди выбранных K сигналов
    input_scale = input_tensor.abs().max()
    if input_scale > 0:
        input_tensor = input_tensor / input_scale
        target_tensor = target_tensor / input_scale

    # Сдвиг target при необходимости
    if delay_d is not None and delay_d != 0:
        target_tensor = torch.roll(target_tensor, -delay_d, dims=-1)

    # -----------------------------
    # Формируем признак мощности
    # pa_feature shape = [1, K, N]
    # -----------------------------
    pa_min = np.min(np.abs(pa_powers))
    pa_max = np.max(np.abs(pa_powers))
    pa_den = pa_max - pa_min

    pa_feature_list = []
    signal_length = input_tensor.shape[-1]
    for pa_pow in pa_powers_sel:
        feat = torch.full(
            (1, 1, signal_length),
            float(np.abs(pa_pow)),
            dtype=dtype,
            device=device
        )
        pa_feature_list.append(feat)

    pa_feature = torch.cat(pa_feature_list, dim=1)  # [1, K, N]

    if pa_den > 0:
        pa_feature = (pa_feature - pa_min) / pa_den
    else:
        pa_feature = torch.zeros_like(pa_feature)

    # Собираем финальный input:
    # input_with_feat shape = [1, 2, K, N]
    #   channel 0 -> входной сигнал
    #   channel 1 -> мощность
    input_with_feat = torch.stack([input_tensor, pa_feature], dim=1)

    # -----------------------------
    # Размер слота
    # -----------------------------
    full_len = input_with_feat.shape[-1]
    assert full_len % slot_num == 0, (
        f"Signal length ({full_len}) must be divisible by slot_num ({slot_num})."
    )

    slot_input_size = full_len // slot_num
    slot_target_size = target_tensor.shape[-1] // slot_num

    # -----------------------------
    # Вспомогательные функции
    # -----------------------------
    def take_slot_range_input(x: torch.Tensor, slots: range) -> torch.Tensor:
        # x: [1, 2, K, N]
        start = slots[0] * slot_input_size
        length = len(slots) * slot_input_size
        return x[..., start:start + length]  # [1, 2, K, L]

    def take_slot_range_target(y: torch.Tensor, slots: range) -> torch.Tensor:
        # y: [1, K, N]
        start = slots[0] * slot_target_size
        length = len(slots) * slot_target_size
        return y[..., start:start + length]  # [1, K, L]

    def make_train_loader(x_subset: torch.Tensor, y_subset: torch.Tensor):
        """
        x_subset: [1, 2, K, L]
        y_subset: [1, K, L]

        Режем каждый сигнал отдельно.
        Не допускаем, чтобы контекст pad_zeros залезал в соседний сигнал.
        """

        # [K, 2, L]
        x_sigs = x_subset[0].permute(1, 0, 2).contiguous()

        # [K, 1, L]
        y_sigs = y_subset[0].unsqueeze(1).contiguous()

        local_block_size = block_size if block_size is not None else x_sigs.shape[-1]

        # ВАЖНО: padding отдельно для каждого сигнала
        x_sigs_pad = F.pad(x_sigs, (pad_zeros, pad_zeros))

        # x_blocks: [K, 2, M, block_size + 2*pad_zeros]
        x_blocks = x_sigs_pad.unfold(
            dimension=-1,
            size=local_block_size + 2 * pad_zeros,
            step=local_block_size
        )

        # y_blocks: [K, 1, M, block_size]
        y_blocks = y_sigs.unfold(
            dimension=-1,
            size=local_block_size,
            step=local_block_size
        )

        M = min(x_blocks.shape[2], y_blocks.shape[2])
        x_blocks = x_blocks[:, :, :M, :]
        y_blocks = y_blocks[:, :, :M, :]

        # [K, 2, M, W] -> [K*M, 2, W]
        x_blocks = x_blocks.permute(0, 2, 1, 3).reshape(
            -1,
            2,
            local_block_size + 2 * pad_zeros
        )

        # [K, 1, M, B] -> [K*M, 1, B]
        y_blocks = y_blocks.permute(0, 2, 1, 3).reshape(
            -1,
            1,
            local_block_size
        )

        ds = (x_blocks, y_blocks)
        ds = ResampleDataset(ds, batch_size=batch_size, aggregate_power="concat")
        ds = torch.utils.data.DataLoader(ds, batch_size=None)

        return ds

    # def make_train_loader(x_subset: torch.Tensor, y_subset: torch.Tensor):
    #     """
    #     x_subset: [1, 2, K, L]
    #     y_subset: [1, K, L]

    #     Нужно:
    #     - склеить K сигналов по времени
    #     - затем unfold по block_size
    #     """
    #     # -> [1, 2, K*L]
    #     x_concat = x_subset.view(1, 2, -1)
    #     # -> [1, 1, K*L]
    #     y_concat = y_subset.view(1, 1, -1)

    #     # block_size по умолчанию = весь train кусок
    #     local_block_size = block_size if block_size is not None else x_concat.shape[-1]

    #     # padding по краям для входа
    #     x_concat = F.pad(x_concat, (pad_zeros, pad_zeros))

    #     # padding для корректного unfold
    #     # step_x = int(local_block_size)
    #     # pad_unfold_input = (step_x + 2 * pad_zeros - x_concat.size(-1) % step_x) % step_x
    #     # x_concat = F.pad(x_concat, (0, pad_unfold_input))

    #     # step_y = int(local_block_size)
    #     # pad_unfold_target = (step_y - y_concat.size(-1) % step_y) % step_y
    #     # y_concat = F.pad(y_concat, (0, pad_unfold_target))

    #     # unfold
    #     # x: [num_blocks, 2, block_size + 2*pad_zeros]
    #     # y: [num_blocks, 1, block_size]
    #     x_blocks = x_concat.unfold(2, local_block_size + 2 * pad_zeros, local_block_size)[0].permute(1, 0, 2)
    #     y_blocks = y_concat.unfold(2, local_block_size, local_block_size)[0].permute(1, 0, 2)

    #     ds = (x_blocks, y_blocks)
    #     ds = ResampleDataset(ds, batch_size=batch_size, aggregate_power="concat")
    #     ds = torch.utils.data.DataLoader(ds, batch_size=None)
    #     return ds

    def make_eval_loader(x_subset: torch.Tensor, y_subset: torch.Tensor):
        """
        x_subset: [1, 2, K, L]
        y_subset: [1, K, L]

        Нужно:
        - НЕ конкатенировать K сигналов
        - вернуть K отдельных сигналов
        """
        # [K, 1, 2, L]
        x_sep = x_subset[0].permute(1, 0, 2)#[:, None, ...]
        # [K, 1, 1, L]
        y_sep = y_subset.permute(1, 0, 2)#[:, None, ...]

        x_sep = F.pad(x_sep, (pad_zeros, pad_zeros))

        ds = (x_sep, y_sep)
        ds = ResampleDataset(ds, batch_size=batch_size, aggregate_power="concat")
        ds = torch.utils.data.DataLoader(ds, batch_size=None)
        return ds

    # -----------------------------
    # Формирование выборок
    # -----------------------------
    train_input_subset = take_slot_range_input(input_with_feat, train_slots_ind)
    train_target_subset = take_slot_range_target(target_tensor, train_slots_ind)

    valid_input_subset = take_slot_range_input(input_with_feat, validat_slots_ind)
    valid_target_subset = take_slot_range_target(target_tensor, validat_slots_ind)

    test_input_subset = take_slot_range_input(input_with_feat, test_slots_ind)
    test_target_subset = take_slot_range_target(target_tensor, test_slots_ind)

    train_set = make_train_loader(train_input_subset, train_target_subset)
    validat_set = make_eval_loader(valid_input_subset, valid_target_subset)
    test_set = make_eval_loader(test_input_subset, test_target_subset)

    dataset = (train_set, validat_set, test_set)

    if return_scales:
        return dataset, [scales, scales_target]
    else:
        return dataset

def dataset_prepare(mat: dict, dtype: torch.dtype = torch.complex128, device: str = 'cuda', batch_size: OptionalInt = None, 
                    block_size: OptionalInt = None, slot_num: OptionalInt = None, pad_zeros: OptionalInt = None, 
                    delay_d: OptionalInt = None, train_slots_ind: range = range(1), validat_slots_ind: range = range(1),
                    test_slots_ind: range = range(1)) -> DatasetType:
    """
    The method extracts input and target data for the mat file, normalizes and resamples if necessary.
    Then it divides input and target tensors into the batches and loads them into the dataloader.

    Args:
        mat (Dictionary): mat-file, which contains:
            input (numpy.ndarray): 1d array with shape (1, arr.shape), which contains input data samples.
            target (numpy.ndarray): 1d array with shape (1, arr.shape), which contains target data samples.
            noise_floor (numpy.ndarray): 1d array with shape (1, arr.shape), which contains noise floor samples.
        dtype (torch.dtype): The type of tensor to convert content of the mat-file to. Defaults is torch.complex128.
        device (str): The device to load dataset to. Defaults is 'cpu'.
        slot_num (int, optional): The number of slots to divide the whole dataset into.
        pad_zeros (int, optional): The number of zeros to add to the beginning and to the end of the signal.
        delay_d (int, optional): The value of desired and noise floor signals shift. delay_d > 0 corresponds to
            samples shift left, delay_d < 0 corresponds to samples shift right. If delay_d equal zero or None, 
            then there is no shift. Defaults is "None".
        train_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
        validat_slots_ind (range): Used only for hold-out cross-validation. Indices of the slots which are chosen for validation dataset. 
            A range with step 1. Defaults is range(1).
        test_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
            
    Returns:
        Tuple of iterables.
    """
    
    if pad_zeros is None:
        pad_zeros = 0

    if batch_size is None:
        batch_size = 1

    input_a = mat['PDinA'][0, :]
    input_b = mat['PDinB'][0, :]
    target_a = mat['PDoutA'][0, :] - mat['PDinA'][0, :]
    target_b = mat['PDoutB'][0, :] - mat['PDinB'][0, :]
    nf = np.zeros_like(target_a)

    if delay_d is not None and delay_d != 0:
        target_a = np.roll(target_a, -delay_d)
        target_b = np.roll(target_b, -delay_d)
        nf = np.roll(nf, -delay_d)

    input_tens_a = torch.tensor(input_a, dtype=dtype).view(1, 1, -1).to(device)
    input_tens_b = torch.tensor(input_b, dtype=dtype).view(1, 1, -1).to(device)
    input = torch.cat((input_tens_a, input_tens_b), dim=1)
    target_tens_a = torch.tensor(target_a, dtype=dtype).view(1, 1, -1).to(device)
    target_tens_b = torch.tensor(target_b, dtype=dtype).view(1, 1, -1).to(device)
    target = torch.cat((target_tens_a, target_tens_b), dim=1)
    nf = torch.tensor(nf, dtype=dtype).view(1, 1, -1).to(device)

    input = input/30000
    target = target/30000
    # alpha = 0.9#/30000
    # scale_input_a = input[:, :1, :].abs().max()
    # scale_input_b = input[:, 1:, :].abs().max()
    # input[:, :1, :] = alpha * (input[:, :1, :].to(device) / scale_input_a)
    # input[:, 1:, :] = alpha * (input[:, 1:, :].to(device) / scale_input_b)
    # scale_target_a = target[:, :1, :].abs().max()
    # scale_target_b = target[:, 1:, :].abs().max()
    # target[:, :1, :] = alpha * (target[:, :1, :].to(device) / scale_target_a)
    # target[:, 1:, :] = alpha * (target[:, 1:, :].to(device) / scale_target_b)

    assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
        "All train slots indices (argument train_slots_ind) must be positive and lower, than number of slots (argument slot_num)."
    assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
        "All validation slots indices (argument validat_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
        "All test slots indices (argument test_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert type(train_slots_ind) == range and type(test_slots_ind) == range and type(validat_slots_ind), \
        f"Types of train, validation and test indices must be a range, but {type(train_slots_ind)}, {type(validat_slots_ind)} and {type(test_slots_ind)} are given correspondingly."
    assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
        f"Step of indices ranges train_slots_ind, validat_slots_ind and test_slots_ind must equal 1, but {train_slots_ind.step}, {validat_slots_ind.step} and {test_slots_ind.step} are given correspondingly."

    slot_input_size = int(input.shape[-1]/slot_num)
    slot_target_size = int(target.shape[-1]/slot_num)
    
    input_train_size = slot_input_size * len(train_slots_ind)
    target_train_size = slot_target_size * len(train_slots_ind)
    input_validat_size = slot_input_size * len(validat_slots_ind)
    target_validat_size = slot_target_size * len(validat_slots_ind)
    input_test_size = slot_input_size * len(test_slots_ind)
    target_test_size = slot_target_size * len(test_slots_ind)
    
    if block_size is None: 
        block_size = input_train_size
    block_size_target = block_size
    
    block_size_test = input_test_size
    block_size_test_target = block_size_test

    block_size_validat = input_validat_size
    block_size_validat_target = block_size_validat
    
    dataset = list()
    
    train_input_set = input[..., train_slots_ind[0] * slot_input_size: train_slots_ind[0] * slot_input_size + input_train_size]
    train_input_set = F.pad(train_input_set, (pad_zeros, pad_zeros))
    train_target_set = torch.cat((target, nf), dim=1)[..., train_slots_ind[0] * slot_target_size: train_slots_ind[0] * slot_target_size + target_train_size]

    train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size))[0, ...].permute(1, 0, 2)
    train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target))[0, ...].permute(1, 0, 2)
    train_set = tuple((train_input_set, train_target_set))
    train_set = ResampleDataset(train_set, batch_size=batch_size)

    train_set = torch.utils.data.DataLoader(train_set, batch_size=None)
    
    validat_input_set = input[..., validat_slots_ind[0] * slot_input_size: validat_slots_ind[0] * slot_input_size + input_validat_size]
    validat_input_set = F.pad(validat_input_set, (pad_zeros, pad_zeros))
    validat_target_set = torch.cat((target, nf), dim=1)[..., validat_slots_ind[0] * slot_target_size: validat_slots_ind[0] * slot_target_size + target_validat_size]
    validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat)[0, ...].permute(1, 0, 2)
    validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target)[0, ...].permute(1, 0, 2)
    validat_set = tuple((validat_input_set, validat_target_set))
    validat_set = ResampleDataset(validat_set, batch_size=batch_size)
    validat_set = torch.utils.data.DataLoader(validat_set, batch_size=None)

    test_input_set = input[..., test_slots_ind[0] * slot_input_size: test_slots_ind[0] * slot_input_size + input_test_size]
    test_input_set = F.pad(test_input_set, (pad_zeros, pad_zeros))
    test_target_set = torch.cat((target, nf), dim=1)[..., test_slots_ind[0] * slot_target_size: test_slots_ind[0] * slot_target_size + target_test_size]
    test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
    test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
    test_set = tuple((test_input_set, test_target_set))
    test_set = ResampleDataset(test_set, batch_size=batch_size)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=None)
    
    dataset.append(tuple((train_set, validat_set, test_set)))
    return dataset[0]