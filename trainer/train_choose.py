import torch
from torch import nn, Tensor
from typing import List, Tuple, Union, Callable, Iterable
from .algorithms import train_sgd_auto, train_mixed_newton_levenb_marq

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]

DataLoaderType = Iterable
StrOrList = Union[str, List[str], Tuple[str], None]
OptionalDataLoader = Union[Iterable, None]
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
OptionalBatchTensor = Union[Callable[[Tensor], Tuple[Tensor, ...]], None]

def train(model: nn.Module, train_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
          config_train: dict, batch_to_tensors: OptionalBatchTensor = None, validate_dataset: OptionalDataLoader = None, 
          test_dataset: OptionalDataLoader = None, train_type: OptionalStr = None,
          save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
          chunk_num: OptionalInt = None, weight_names: StrOrList = None, device: OptionalStr = None) -> None:
    """
    This function activates model training functions depending on the required training type.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (Iterable): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to train model on.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
            process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        config_train (dictionary): Dictionary with configurations of training procedure. Includes learning rate, training type,
            optimizers parameters etc. Implied to be loaded from .yaml config file.
        batch_to_tensors (Callable, optional): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
            Attention! batch_to_tensors argumnet is only important in case train_type == 'mnm', in other cases it could be omitted.
        validate_dataset (Iterable, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
            Defaults is "None".
        test_dataset (Iterable, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
            Defaults is "None".
        train_type - flag, that shows which algorithm to exploit in training.
            train_type == 'mnm_lev_marq', - corresponds to Levenberg–Marquardt algorithm on base of Mixed Newton Method,
            train_type == 'sgd_auto', - corresponds to Stochastic Gradient Descent, implemented by means of loss.backward() function.
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        chunk_num(int, optional): The number of chunks in dataset. Defaults to "None".
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".
        device (str, optional): device to implement calculations: "cpu" or "cuda:0". Defaults is None.

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    if save_path is None:
        save_path = ''
    else:
        save_path += '/'
    if exp_name is None:
        exp_name = ''
    else:
        exp_name = '_' + exp_name

    if train_type is None:
        train_type = 'sgd_auto'

    best_criterion = None

    save_signals = True
    
    torch.save(model.state_dict(), save_path+'weights_init'+exp_name)

    if train_type == 'sgd_auto':
        learning_curve, best_criterion = train_sgd_auto(model, train_dataset, validate_dataset, test_dataset, loss_fn, 
                                                        quality_criterion, batch_to_tensors, config_train, save_path, exp_name,
                                                        save_every, weight_names)
    elif train_type == 'mnm_lev_marq':
        learning_curve, best_criterion = train_mixed_newton_levenb_marq(model, train_dataset, validate_dataset, test_dataset, loss_fn, 
                                                                        quality_criterion, batch_to_tensors, chunk_num, 
                                                                        save_path, exp_name, save_every, save_signals, weight_names)
    else:
        print(f"Attention! Training type \'{train_type}\' doesn`t match any of the possible types: \'sgd\', \'mnm\'.")
    return learning_curve, best_criterion