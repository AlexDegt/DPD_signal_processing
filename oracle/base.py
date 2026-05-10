import torch
from torch import nn, Tensor
from functools import reduce
# from torch.func import jacrev
from utils import Timer
from math import ceil
import copy
import sys

from typing import List, Tuple, Union, Callable, Iterable

LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchType = Union[Tensor, Tuple[Tensor, ...]]
TensorTuple = Tuple[Tensor, ...]
DerRetType = Union[Tuple[Tensor, ...], Tensor]
StrOrList = Union[str, List[str], Tuple[str], None]
OptionalTensor = Union[Tensor, None]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]
OptionalInt = Union[int, None]


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    for i in range(len(names[:-1])):
        obj = getattr(obj, names[i], None)
        if obj is None:
            return
    if hasattr(obj, names[-1]):
        delattr(obj, names[-1])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor, is_nn_param: bool = False) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _set_nested_attr(obj, ['conv', 'weight'], value)
    """
    for i in range(len(names[:-1])):
        obj = getattr(obj, names[i], None)
        if obj is None:
            return
    if is_nn_param:
        setattr(obj, names[-1], nn.Parameter(value))
    else:
        setattr(obj, names[-1], value)


def count_parameters(model: nn.Module, count_non_differentiable: bool = True) -> int:
    """
    This function counts the overall number of scalar parameters in the model.
    
    Args:
        model (nn.Module): The model with parameters to count.
        count_non_differentiable (bool, optional): If set "True" the whole set of parameters is affected.
            If set "False" only differentiable parameters are counted. Defaults to "True".
    
    Returns:
        int: The return value. The number of parametrs the model stores.
    """
    if count_non_differentiable:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_weights(mod: nn.Module, name_list: StrOrList = None) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    By spceifying `name_list` it is possible to extract only several named parameters.
    """
    if name_list is None:
        orig_params = tuple(mod.parameters())
        # Remove all the parameters in the model
        names = []
        for name, p in list(mod.named_parameters()):
            _del_nested_attr(mod, name.split("."))
            names.append(name)
        
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in orig_params)
        return params, names
    if isinstance(name_list, str):
        name_list = [name_list]
    params = tuple(reduce(getattr, name.split(sep='.'), mod).detach().requires_grad_() for name in name_list)
    for name in name_list:
        _del_nested_attr(mod, name.split("."))
    return params, name_list


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...], is_nn_param: bool = False) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p, is_nn_param)


def _check_tensors_complex_any(tensors):
    """
    Checks input for containment of any complex-valued tensor.
    Args:
        tensors (Tensor, list of Tensor, tuple of Tensor or other iterable of Tensor): A bunch of tensors to check for
            containing any complex number.
    """
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for t in tensors:
        if torch.is_complex(t):
            return True
    return False


class Oracle(object):
    """
    This class works as wrapper around model with only differentiable parameters. Takes model instance
    and loss function which evaluates the model quality. The class supports interaction with parameters,
    view them as 1d Torch tensors, gets and sets parameter values. The Oracle computes loss function value,
    gradient in two ways, hessian.
    
    Attributes:
        _model (nn.Module): The model with differentiable parameters.
        _loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar.
    """
    
    def __init__(self, model: nn.Module, loss_fn: LossFnType, inplace_copy_model: bool = False) -> None:
        """
        Class constructor.
        
        Args:
            model (nn.Module): The model with differentiable parameters.
            loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar.
            inplace_copy_model (bool, optional): This flag defines whether to perform deep copy of the model
                or to use the same referenced instance. Defaults to "False".
        """
        if inplace_copy_model:
            self._model = copy.deepcopy(model)
        else:
            self._model = model
        self._loss_fn = loss_fn
    
    def get_params_names(self) -> List[str]:
        """
        Returns names of parameters for reference in other class methods.
        
        Returns:
            List of str.
        """
        names = []
        for name, _ in self._model.named_parameters():
            names.append(name)
        return names
    
    def get_flat_params(self, detach: bool = True, name_list: StrOrList = None, idxs: OptionalTensor = None) -> Tensor:
        """
        Use this method to get the model parameters copy in 1d vector form.
        
        Args:
            detach (bool, optional): If equals "False", the result 1d Tensor is treated as differentiable
                vector-function of nn.Parameter instances. Defaults to "True".
            name_list (str or list of str, optional): By spceifying `name_list` it is possible to get only several named
                parameters concatenated into 1d vector. Defaults to "None".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original 1d flattened vector of model parameters. If idxs is not None we consider returned vector
                as a 1d float Tensor of size len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original 1d flattened vector of model parameters. If name_list is specified, indexes from idxs represent
                positions from resulting 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
        
        Returns:
            Tensor: 1d Tensor of the model parameters.
        """
        views = []
        if name_list is None:
            for p in self._model.parameters():
                if detach:
                    views.append(p.detach().view(-1))
                else:
                    views.append(p.view(-1))
            if idxs is None:
                return torch.cat(views, 0)
            return torch.cat(views, 0)[idxs]
        if isinstance(name_list, str):
            name_list = [name_list]
        for name in name_list:
            if detach:
                views.append(reduce(getattr, name.split(sep='.'), self._model).detach().view(-1))
            else:
                views.append(reduce(getattr, name.split(sep='.'), self._model).view(-1))
        if idxs is None:
            return torch.cat(views, 0)
        return torch.cat(views, 0)[idxs]
    
    def _model_output(self, signal_batch: BatchType) -> Tensor:
        """
        This method computes model output and returns differentiable Tensor.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
        
        Returns:
            complex Tensor: The model output value. This value is differentiable.
        """
        model_input = self._batch_to_tensors(signal_batch)[0]
        return self._model(model_input)

    @torch.no_grad()
    def set_flat_params(self, flat_params: Tensor, name_list: StrOrList = None, idxs: OptionalTensor = None) -> None:
        """
        Use this method to set the model parameters from 1d vector form.
        
        Args:
            flat_params (Tensor): 1d float Tensor used to store model parameters.
            name_list (str or list of str, optional): By spceifying `name_list` it is possible to set only several named
                parameters from 1d vector. It is strictly advised to have a similar order of named parameters used in
                get_flat_params method due to model consistency reasons. Defaults to "None".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked update only for specified indexes in
                flat_params. If idxs is not None we consider flat_params as a 1d float Tensor of size len(idxs) with elements
                corresponding to indexes from idxs representing positions in the original 1d flattened vector of model
                parameters. If name_list is specified, indexes from idxs represent positions from 1d flattened vector of model
                parameters selected wrt name_list. Defaults to "None".
        """
        if not (idxs is None):
            old_params = self.get_flat_params(detach=True, name_list=name_list)
            old_params[idxs] = flat_params
            flat_params = old_params
        offset = 0
        if name_list is None:
            for p in self._model.parameters():
                numel = p.numel()
                # view as to avoid deprecated pointwise semantics
                p.copy_(flat_params[offset:offset + numel].view_as(p))
                offset += numel
        else:
            if isinstance(name_list, str):
                name_list = [name_list]
            for name in name_list:
                p = reduce(getattr, name.split(sep='.'), self._model)
                numel = p.numel()
                p.copy_(flat_params[offset:offset + numel].view_as(p))
                offset += numel
    
    @torch.no_grad()
    def loss_function_val(self, signal_batch: BatchType) -> Tensor:
        """
        This method computes loss function value and returns nondifferentiable scalar Tensor.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
        
        Returns:
            float scalar Tensor: The loss function value. This value is nondifferentiable.
        """
        return self._loss_fn(self._model, signal_batch)

    @torch.enable_grad()
    def direction_through_jacobian(self, signal_batch: BatchType, batch_to_tensors: BatchTensorType,
                    weight_names: StrOrList = None, compute_fn_val: bool = False, 
                    return_full_wirtinger_derivative: bool = False, idxs: OptionalTensor = None) -> DerRetType:
        """
        This method computes hessian and gradient values of the loss function and optionally returns loss function value.
        The method accumulates jacobian from the jacobian chunks generated by model_output_jacobian_chunk function. 
        The gradient is computed using model output (1d tensor) jacobian w.r.t. the model parameters: grad = jacobian^H @ (model_output - target).
        The hessian is computed also using model output jacobian w.r.t. the model parameters: hess = jacobian^H @ jacobian.
        Generated jacobian is batched: jacobian.size() = [batch_size, sample_size, model_parameter_number].
        Current function is intended to be used only for holomorphic functions. The method makes .zero_grad() for inner _model.
        BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality on.
            batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
                the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
                for several named parameters. Defaults to "None".
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a gradient wrt (z, z*) variables is
                returned, by default only d / dz* is returned. Defaults to "False".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original gradient. If idxs is not None we consider returned gradient as a 1d float Tensor of size
                len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original gradient wrt model parameters. For return_full_wirtinger_derivative=True a returned
                1d float Tensor has size [2 * len(idxs)]. If weight_names is specified, indexes from idxs
                represent positions from 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
                
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: hessian.
            Tensor: gradient.
        """

        if isinstance(signal_batch, Tensor):
            sample_size = signal_batch.size()[-1]
            batch_size = tuple(signal_batch.size())[0]
            resample_ratio = 1

        if isinstance(signal_batch, Iterable):
            sample_size = signal_batch[0].size()[-1]
            batch_size = tuple(signal_batch[0].size())[0]
            resample_ratio = int(sample_size/signal_batch[1].size()[-1])

        if sample_size >= count_parameters(self._model):
            vectorize = True
            strategy = 'forward-mode'
        else:
            vectorize = False
            strategy = 'reverse-mode'

        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)

        signal_batch_input, signal_batch_output = batch_to_tensors(signal_batch)[:2]

        self._model.zero_grad()

        if weight_names is None:
            params = tuple(self._model.parameters())
        else:
            if isinstance(weight_names, str):
                weight_names = [weight_names]

            params = tuple(reduce(getattr, name.split(sep='.'), self._model) for name in weight_names)

        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model, weight_names)

        if _check_tensors_complex_any(params):# z = x + i * y
            real_params = tuple(t.real for t in params)
            num_real_params = len(real_params)
            imag_params = tuple(t.imag for t in params)
            joint_params = (*real_params, *imag_params)

            def f_x_y(*joint_weights):
                weights = tuple(
                    re + 1.j * im for re, im in zip(joint_weights[:num_real_params], joint_weights[num_real_params:]))
                load_weights(self._model, names, weights)
                return self._model(signal_batch_input)

            J = torch.autograd.functional.jacobian(f_x_y, tuple(joint_params), create_graph=False, vectorize=vectorize, strategy=strategy)

            J = tuple(j.view(batch_size, j.size()[2], -1) for j in J)

            J_x = torch.cat(tuple(j for j in J[:num_real_params]), dim=2)
            J_y = torch.cat(tuple(j for j in J[num_real_params:]), dim=2)

            if return_full_wirtinger_derivative:
                J_z = (J_x - 1.j * J_y) / 2.
                J_z_conj = (J_x + 1.j * J_y) / 2.
                if idxs is None:
                    J = torch.cat((J_z, J_z_conj), dim=2)
                else:
                    J = torch.cat((J_z[..., idxs], J_z_conj[..., idxs]), dim=2)
            else:
                if idxs is None:
                    J = (J_x - 1.j * J_y) / 2.
                else:
                    J = ((J_x - 1.j * J_y) / 2.)[..., idxs]
        else:
            def f(*weights):
                load_weights(self._model, names, weights)
                return self._model(signal_batch_input)
            
            J = torch.autograd.functional.jacobian(f, tuple(params), create_graph=False, vectorize=vectorize, strategy=strategy)
            if idxs is None:
                J = torch.cat(tuple(j.view(batch_size, j.size()[2], -1) for j in J), dim=2)
            else:
                J = torch.cat(tuple(j.view(batch_size, j.size()[2], -1) for j in J), dim=2)[..., idxs]

        load_weights(self._model, names, params, is_nn_param=True)

        J_H = torch.conj(torch.permute(J, (0, 2, 1)))            
        
        model_output = self._model(signal_batch_input)

        error_vec = torch.permute(model_output - signal_batch_output, (0, 2, 1))

        grad = torch.bmm(J_H, error_vec)
        hess = torch.bmm(J_H, J)

        grad = torch.sum(grad, keepdim=False, dim=0)
        hess = torch.sum(hess, keepdim=False, dim=0)

        grad.detach_()
        hess.detach_()
        J = J.detach()
        J_H = J_H.detach()
        error_vec = error_vec.detach()
        del J
        del J_H
        del error_vec

        torch.cuda.empty_cache()
        
        if compute_fn_val:
            return loss_val, hess, grad.view(-1)
        return hess, grad.view(-1)

    @torch.enable_grad()
    def gradient_holom_func(self, signal_batch: BatchType, batch_to_tensors: BatchTensorType,
                    weight_names: StrOrList = None, compute_fn_val: bool = False, 
                    return_full_wirtinger_derivative: bool = False, idxs: OptionalTensor = None) -> DerRetType:
        """
        This method computes gradient values of the loss function and optionally returns loss function value.
        The method accumulates jacobian from the jacobian chunks generated by model_output_jacobian_chunk function. 
        The gradient is computed using model output (1d tensor) jacobian w.r.t. the model parameters: grad = jacobian^H @ (model_output - target).
        Generated jacobian is batched: jacobian.size() = [batch_size, sample_size, model_parameter_number].
        Current function is intended to be used only for holomorphic functions. The method makes .zero_grad() for inner _model.
        BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality on.
            batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
                the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
                for several named parameters. Defaults to "None".
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a gradient wrt (z, z*) variables is
                returned, by default only d / dz* is returned. Defaults to "False".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original gradient. If idxs is not None we consider returned gradient as a 1d float Tensor of size
                len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original gradient wrt model parameters. For return_full_wirtinger_derivative=True a returned
                1d float Tensor has size [2 * len(idxs)]. If weight_names is specified, indexes from idxs
                represent positions from 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
                
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: hessian.
            Tensor: gradient.
        """

        if isinstance(signal_batch, Tensor):
            sample_size = signal_batch.size()[-1]
            batch_size = tuple(signal_batch.size())[0]
            resample_ratio = 1

        if isinstance(signal_batch, Iterable):
            sample_size = signal_batch[0].size()[-1]
            batch_size = tuple(signal_batch[0].size())[0]
            resample_ratio = int(sample_size/signal_batch[1].size()[-1])

        if sample_size >= count_parameters(self._model):
            vectorize = True
            strategy = 'forward-mode'
        else:
            vectorize = False
            strategy = 'reverse-mode'

        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)

        signal_batch_input, signal_batch_output = batch_to_tensors(signal_batch)[:2]

        self._model.zero_grad()

        if weight_names is None:
            params = tuple(self._model.parameters())
        else:
            if isinstance(weight_names, str):
                weight_names = [weight_names]

            params = tuple(reduce(getattr, name.split(sep='.'), self._model) for name in weight_names)

        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model, weight_names)

        if _check_tensors_complex_any(params):# z = x + i * y
            real_params = tuple(t.real for t in params)
            num_real_params = len(real_params)
            imag_params = tuple(t.imag for t in params)
            joint_params = (*real_params, *imag_params)

            def f_x_y(*joint_weights):
                weights = tuple(
                    re + 1.j * im for re, im in zip(joint_weights[:num_real_params], joint_weights[num_real_params:]))
                load_weights(self._model, names, weights)
                return self._model(signal_batch_input)

            J = torch.autograd.functional.jacobian(f_x_y, tuple(joint_params), create_graph=False, vectorize=vectorize, strategy=strategy)

            J = tuple(j.view(batch_size, j.size()[2], -1) for j in J)

            J_x = torch.cat(tuple(j for j in J[:num_real_params]), dim=2)
            J_y = torch.cat(tuple(j for j in J[num_real_params:]), dim=2)

            if return_full_wirtinger_derivative:
                J_z = (J_x - 1.j * J_y) / 2.
                J_z_conj = (J_x + 1.j * J_y) / 2.
                if idxs is None:
                    J = torch.cat((J_z, J_z_conj), dim=2)
                else:
                    J = torch.cat((J_z[..., idxs], J_z_conj[..., idxs]), dim=2)
            else:
                if idxs is None:
                    J = (J_x - 1.j * J_y) / 2.
                else:
                    J = ((J_x - 1.j * J_y) / 2.)[..., idxs]
        else:
            def f(*weights):
                load_weights(self._model, names, weights)
                return self._model(signal_batch_input)
            
            J = torch.autograd.functional.jacobian(f, tuple(params), create_graph=False, vectorize=vectorize, strategy=strategy)
            if idxs is None:
                J = torch.cat(tuple(j.view(batch_size, j.size()[2], -1) for j in J), dim=2)
            else:
                J = torch.cat(tuple(j.view(batch_size, j.size()[2], -1) for j in J), dim=2)[..., idxs]

        load_weights(self._model, names, params, is_nn_param=True)            
        
        model_output = self._model(signal_batch_input)

        error_vec = torch.permute(model_output - signal_batch_output, (0, 2, 1))

        grad = torch.bmm(torch.conj(torch.permute(J, (0, 2, 1))), error_vec)

        grad = torch.sum(grad, keepdim=False, dim=0)

        grad.detach_()
        J = J.detach()
        error_vec = error_vec.detach()
        del J
        del error_vec

        torch.cuda.empty_cache()
        
        if compute_fn_val:
            return loss_val, grad.view(-1)
        return grad.view(-1) 

    @torch.enable_grad()
    def hessian(self, signal_batch: BatchType, weight_names: StrOrList = None, compute_fn_val: bool = False,
                outer_jacobian_strategy: str = "reverse-mode", return_full_wirtinger_derivative: bool = False,
                idxs: OptionalTensor = None, return_for_cubic_newton = True, vectorize: bool = False,
                return_full_real_derivative: bool = False) -> DerRetType:
        """
        This method computes hessian value of the loss function and optionally returns loss function value
        as nondifferentiable scalar. The hessian is computed using torch.autograd.functional.hessian function. This method
        makes .zero_grad() for inner _model.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute hessian only
                for several named parameters. Defaults to "None".
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            outer_jacobian_strategy (str, optional): The Hessian is computed by computing the Jacobian of a Jacobian. The inner
                Jacobian is always computed in reverse-mode AD. Setting strategy to "forward-mode" or "reverse-mode" determines
                whether the outer Jacobian will be computed with forward or reverse mode AD.
                Currently, computing the outer Jacobian in "forward-mode" requires vectorized=True. Defaults to "reverse-mode".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a hessian wrt (z, z*) variables is returned,
                by default only d^2 / (dz*dz) is returned. Defaults to "False".
            return_full_real_derivative (bool, optional): If specified "True" a hessian wrt (Re(z), Im(z)) variables is returned,
                by default only d^2 / (dz*dz) is returned. Defaults to "False".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original hessian. If idxs is not None we consider returned hessian as a 2d float Tensor of shape
                len(idxs) x len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original hessian wrt model parameters. For return_full_wirtinger_derivative=True a returned
                2d float Tensor has shape [2 * len(idxs)] x [2 * len(idxs)]. If weight_names is specified, indexes from idxs
                represent positions from 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: The hessian value.
        """
        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model, weight_names)
        
        def _reshape_param_tensor_tuple(row_id, tensor_tuple):
            row_size = params[row_id].numel()
            reshaped_tensor_list = []
            for col_id, tensor in enumerate(tensor_tuple):
                col_size = params[col_id].numel()
                reshaped_tensor_list.append(tensor_tuple[col_id].view(row_size, col_size))
            return tuple(t for t in reshaped_tensor_list)
        
        if _check_tensors_complex_any(params):# z = x + i * y
            real_params = tuple(t.real for t in params)
            num_real_params = len(real_params)
            imag_params = tuple(t.imag for t in params)
            joint_params = (*real_params, *imag_params)
            
            def f_xx_xy_yy(*joint_weights):
                weights = tuple(
                    re + 1.j * im for re, im in zip(joint_weights[:num_real_params], joint_weights[num_real_params:]))
                load_weights(self._model, names, weights)
                return self._loss_fn(self._model, signal_batch)
            
            H_res = torch.autograd.functional.hessian(f_xx_xy_yy, tuple(joint_params), create_graph=False, vectorize=vectorize,
                                                      outer_jacobian_strategy=outer_jacobian_strategy)

            H_res_xx = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row[:num_real_params]), dim=1)\
                                  for row, h_row in enumerate(H_res[:num_real_params])), dim=0)
            H_res_xy = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row[num_real_params:]), dim=1)\
                                  for row, h_row in enumerate(H_res[:num_real_params])), dim=0)
            H_res_yx = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row[:num_real_params]), dim=1)\
                                  for row, h_row in enumerate(H_res[num_real_params:])), dim=0)
            if return_for_cubic_newton:
                H = (2 * H_res_xx + 1.0j * (H_res_yx - H_res_xy)) / 4.0
                if idxs is not None:
                    H = H[idxs[:, None], idxs]
            else:
                H_res_yy = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row[num_real_params:]), dim=1) \
                                           for row, h_row in enumerate(H_res[num_real_params:])), dim=0)
            if return_full_real_derivative:
                if idxs is None:
                    H = torch.cat((torch.cat((H_res_xx, H_res_xy), dim=1), torch.cat((H_res_yx, H_res_yy), dim=1)),
                                  dim=0)
                else:
                    H = torch.cat((torch.cat((H_res_xx[idxs[:, None], idxs], H_res_xy[idxs[:, None], idxs]), dim=1),
                                   torch.cat((H_res_yx[idxs[:, None], idxs], H_res_yy[idxs[:, None], idxs]), dim=1)),
                                  dim=0)
            elif return_full_wirtinger_derivative:
                H_z_z = (H_res_xx - H_res_yy - 1.j * (H_res_yx + H_res_xy)) / 4.
                H_z_z_conj = (H_res_xx + H_res_yy + 1.j * (H_res_xy - H_res_yx)) / 4.
                H_z_conj_z = (H_res_xx + H_res_yy + 1.j * (H_res_yx - H_res_xy)) / 4.
                H_z_conj_z_conj = (H_res_xx - H_res_yy + 1.j * (H_res_yx + H_res_xy)) / 4.
                if idxs is None:
                    H = torch.cat((torch.cat((H_z_z, H_z_z_conj), dim=1), torch.cat((H_z_conj_z, H_z_conj_z_conj), dim=1)),
                                  dim=0)
                else:
                    H = torch.cat((torch.cat((H_z_z[idxs[:, None], idxs], H_z_z_conj[idxs[:, None], idxs]), dim=1),
                                   torch.cat((H_z_conj_z[idxs[:, None], idxs], H_z_conj_z_conj[idxs[:, None], idxs]), dim=1)),
                                  dim=0)
            elif not return_for_cubic_newton:
                if idxs is None:
                    H = (H_res_xx + H_res_yy + 1.j * (H_res_yx - H_res_xy)) / 4.
                else:
                    H = ((H_res_xx + H_res_yy + 1.j * (H_res_yx - H_res_xy)) / 4.)[idxs[:, None], idxs]
        else:
            def f(*weights):
                load_weights(self._model, names, weights)
                return self._loss_fn(self._model, signal_batch)
        
            H_res = torch.autograd.functional.hessian(f, tuple(params), create_graph=False, vectorize=vectorize,
                                                      outer_jacobian_strategy=outer_jacobian_strategy)
            if idxs is None:
                H = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row), dim=1)\
                              for row, h_row in enumerate(H_res)), dim=0)
            else:
                H = torch.cat(tuple(torch.cat(_reshape_param_tensor_tuple(row, h_row), dim=1)\
                              for row, h_row in enumerate(H_res)), dim=0)[idxs[:, None], idxs]
            
        load_weights(self._model, names, params, is_nn_param=True)
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, H
        return H
    
    @torch.enable_grad()
    def gradient_through_jacobian(self, signal_batch: BatchType, weight_names: StrOrList = None,
                                  compute_fn_val: bool = False, return_full_wirtinger_derivative: bool = False,
                                  idxs: OptionalTensor = None, strategy: str = "reverse-mode", vectorize: bool = False,
                                  return_full_real_derivative: bool = False) -> DerRetType:
        """
        This method computes gradient value of the loss function and optionally returns loss function value
        as nondifferentiable scalar. The gradient is computed using torch.autograd.functional.jacobian function. This method
        makes .zero_grad() for inner _model.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
                for several named parameters. Defaults to "None".
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a gradient wrt (z, z*) variables is
                returned, by default only d / dz* is returned. Defaults to "False".
            return_full_real_derivative (bool, optional): If specified "True" a hessian wrt (Re(z), Im(z)) variables is returned,
                by default only d^2 / (dz*dz) is returned. Defaults to "False".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original gradient. If idxs is not None we consider returned gradient as a 1d float Tensor of size
                len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original gradient wrt model parameters. For return_full_wirtinger_derivative=True a returned
                1d float Tensor has size [2 * len(idxs)]. If weight_names is specified, indexes from idxs
                represent positions from 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: The gradient value.
        """
        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model, weight_names)

        if _check_tensors_complex_any(params):# z = x + i * y
            real_params = tuple(t.real for t in params)
            num_real_params = len(real_params)
            imag_params = tuple(t.imag for t in params)
            joint_params = (*real_params, *imag_params)
            
            def f_x_y(*joint_weights):
                weights = tuple(
                    re + 1.j * im for re, im in zip(joint_weights[:num_real_params], joint_weights[num_real_params:]))
                load_weights(self._model, names, weights)
                return self._loss_fn(self._model, signal_batch)

            J = torch.autograd.functional.jacobian(f_x_y, tuple(joint_params), strategy=strategy, create_graph=False, vectorize=vectorize)


            J_x = torch.cat(tuple(j.contiguous().view(1, -1) for j in J[:num_real_params]), dim=1)
            J_y = torch.cat(tuple(j.contiguous().view(1, -1) for j in J[num_real_params:]), dim=1)

            if return_full_real_derivative:
                if idxs is None:
                    J = torch.cat((J_x, J_y), dim=1)
                else:
                    J = torch.cat((J_x[:, idxs], J_y[:, idxs]), dim=1)
            elif return_full_wirtinger_derivative:
                J_z = (J_x - 1.j * J_y) / 2.
                J_z_conj = (J_x + 1.j * J_y) / 2.
                if idxs is None:
                    J = torch.cat((J_z, J_z_conj), dim=1)
                else:
                    J = torch.cat((J_z[:, idxs], J_z_conj[:, idxs]), dim=1)
            else:
                if idxs is None:
                    J = (J_x + 1.j * J_y) / 2.
                else:
                    J = ((J_x + 1.j * J_y) / 2.)[:, idxs]
        else:
            def f(*weights):
                load_weights(self._model, names, weights)
                return self._loss_fn(self._model, signal_batch)
            
            J = torch.autograd.functional.jacobian(f, tuple(params), strategy=strategy, create_graph=False, vectorize=vectorize)
            if idxs is None:
                J = torch.cat(tuple(j.contiguous().view(1, -1) for j in J), dim=1)
            else:
                J = torch.cat(tuple(j.contiguous().view(1, -1) for j in J), dim=1)[:, idxs]

        load_weights(self._model, names, params, is_nn_param=True)
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, J.view(-1)
        return J.view(-1)
    
    @torch.enable_grad()
    def gradient(self, signal_batch: BatchType, weight_names: StrOrList = None, compute_fn_val: bool = False,
                 return_full_wirtinger_derivative: bool = False, idxs: OptionalTensor = None,
                 strategy: str = "reverse-mode", vectorize: bool = False, return_full_real_derivative: bool = False) -> DerRetType:
        """
        This method computes gradient value of the loss function and optionally returns loss function value.
        The gradient is computed using .backward() method called on loss function value. This method makes .zero_grad() for
        inner _model.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
                for several named parameters. Defaults to "None".
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a gradient wrt (z, z*) variables is
                returned, by default only d / dz* is returned. Defaults to "False".
            return_full_real_derivative (bool, optional): If specified "True" a hessian wrt (Re(z), Im(z)) variables is returned,
                by default only d^2 / (dz*dz) is returned. Defaults to "False".
            idxs (Tensor, optional): 1d int Tensor of indexes to perform masked get only for specified indexes from
                the original gradient. If idxs is not None we consider returned gradient as a 1d float Tensor of size
                len(idxs) with elements corresponding to indexes from idxs representing positions
                in the original gradient wrt model parameters. For return_full_wirtinger_derivative=True a returned
                1d float Tensor has size [2 * len(idxs)]. If weight_names is specified, indexes from idxs
                represent positions from 1d flattened vector of model parameters selected wrt name_list. Defaults to "None".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is differentiable.
            Tensor: The gradient value.
        """
        self._model.zero_grad()
        views = []

        if weight_names is None:
            params = tuple(self._model.parameters())
        else:
            if isinstance(weight_names, str):
                weight_names = [weight_names]

            params = tuple(reduce(getattr, name.split(sep='.'), self._model) for name in weight_names)

        if _check_tensors_complex_any(params):# z = x + i * y
            return self.gradient_through_jacobian(signal_batch=signal_batch, weight_names=weight_names,
                                                  compute_fn_val=compute_fn_val,
                                                  return_full_wirtinger_derivative=return_full_wirtinger_derivative, idxs=idxs,
                                                  return_full_real_derivative=return_full_real_derivative,
                                                  strategy=strategy, vectorize=vectorize)

        loss_val = self._loss_fn(self._model, signal_batch)
        loss_val.backward()
        
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        if idxs is None:
            flat_grad = torch.cat(views, dim=0)
        else:
            flat_grad = torch.cat(views, dim=0)[idxs]
        
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, flat_grad
        return flat_grad

    @torch.no_grad()
    def grad_num_check(self, signal_batch: BatchType, weight_names: StrOrList = None,
                       return_full_wirtinger_derivative: bool = False, eps=1e-6, order=2):
        """
        This method estimates correctness of the gradient computation procedure by comparison of the automatically computed
        gradient and numerically estimated gradient using a second-order numerical difference scheme.

        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
                for several named parameters. Defaults to "None".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a gradient wrt (z, z*) variables is
                estimated, by default only d / dz* is estimated. Defaults to "False".
            eps (float, strictly positive): A tolerance value to estimate gradient. Defaults to 1e-6.
            order (int, equals 1 or 2): An order of the approximation scheme. Defaults to 2.
        
        Returns:
            float scalar: a discrepancy between automatically computed gradient and numerically estimated one
                based on infinity norm.
        """
        if order == 1:
            func_val, autograd = self.gradient(signal_batch=signal_batch, weight_names=weight_names, compute_fn_val=True,
                return_full_wirtinger_derivative=return_full_wirtinger_derivative)
        else:
            autograd = self.gradient(signal_batch=signal_batch, weight_names=weight_names, compute_fn_val=False,
                return_full_wirtinger_derivative=return_full_wirtinger_derivative)
        flat_params = self.get_flat_params(detach=True, name_list=weight_names)
        complex_derivative = False
        if _check_tensors_complex_any(flat_params):# z = x + i * y
            complex_derivative = True
            numgrad_x, numgrad_y = torch.zeros_like(flat_params.real), torch.zeros_like(flat_params.imag)
        else:
            numgrad = torch.zeros_like(flat_params)
        
        def evaluate_func_at_point(x):
            self.set_flat_params(flat_params=x, name_list=weight_names)
            return self.loss_function_val(signal_batch=signal_batch)

        delta = torch.zeros_like(flat_params)
        for i in range(numgrad_x.numel()):
            delta[i - 1] = 0.
            delta[i] = 1.
            if complex_derivative:
                if order == 1:
                    numgrad_x[i] = (evaluate_func_at_point(flat_params + eps * delta) - func_val) / eps
                    numgrad_y[i] = (evaluate_func_at_point(flat_params + 1.j * eps * delta) - func_val) / eps
                else:
                    numgrad_x[i] = (evaluate_func_at_point(flat_params + eps * delta) -
                        evaluate_func_at_point(flat_params - eps * delta)) / (2 * eps)
                    numgrad_y[i] = (evaluate_func_at_point(flat_params + 1.j * eps * delta) -
                        evaluate_func_at_point(flat_params - 1.j * eps * delta)) / (2 * eps)
            else:
                if order == 1:
                    numgrad[i] = (evaluate_func_at_point(flat_params + eps * delta) - func_val) / eps
                else:
                    numgrad[i] = (evaluate_func_at_point(flat_params + eps * delta) -
                        evaluate_func_at_point(flat_params - eps * delta)) / (2 * eps)

        if complex_derivative:
            if return_full_wirtinger_derivative:
                numgrad = torch.cat(((numgrad_x - 1.j * numgrad_y) / 2., (numgrad_x + 1.j * numgrad_y) / 2.), dim=0)
            else:
                numgrad = (numgrad_x + 1.j * numgrad_y) / 2.

        self.set_flat_params(flat_params=flat_params, name_list=weight_names)
        return torch.linalg.norm(autograd - numgrad, ord=float("inf")).item()

    @torch.no_grad()
    def hess_num_check(self, signal_batch: BatchType, weight_names: StrOrList = None,
                       outer_jacobian_strategy: str = "reverse-mode", return_full_wirtinger_derivative: bool = False, eps=1e-4,
                       order=2):
        """
        This method estimates correctness of the hessian computation procedure by comparison of the automatically computed
        hessian and numerically estimated hessian using a first-order or second-order numerical difference scheme.

        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute hessian only
                for several named parameters. Defaults to "None".
            outer_jacobian_strategy (str, optional): The Hessian is computed by computing the Jacobian of a Jacobian. The inner
                Jacobian is always computed in reverse-mode AD. Setting strategy to "forward-mode" or "reverse-mode" determines
                whether the outer Jacobian will be computed with forward or reverse mode AD.
                Currently, computing the outer Jacobian in "forward-mode" requires vectorized=True. Defaults to "reverse-mode".
            return_full_wirtinger_derivative (bool, optional): If specified "True" a hessian wrt (z, z*) variables is returned,
                by default only d^2 / (dz*dz) is returned. Defaults to "False".
            eps (float, strictly positive): A tolerance value to estimate hessian. Defaults to 1e-4.
            order (int, equals 1 or 2): An order of the approximation scheme. Defaults to 2.
        
        Returns:
            float scalar: a discrepancy between automatically computed hessian and numerically estimated one
                based on infinity norm.
        """
        if order == 1:
            func_val, autohess = self.hessian(signal_batch=signal_batch, weight_names=weight_names, compute_fn_val=True, 
                outer_jacobian_strategy=outer_jacobian_strategy,
                return_full_wirtinger_derivative=return_full_wirtinger_derivative)
        else:
            autohess = self.hessian(signal_batch=signal_batch, weight_names=weight_names, compute_fn_val=False,
                outer_jacobian_strategy=outer_jacobian_strategy,
                return_full_wirtinger_derivative=return_full_wirtinger_derivative)
        flat_params = self.get_flat_params(detach=True, name_list=weight_names)
        complex_derivative = False
        if _check_tensors_complex_any(flat_params):# z = x + i * y
            complex_derivative = True
            if return_full_wirtinger_derivative:
                n = autohess.shape[0] // 2
            else:
                n = autohess.shape[0]
            numhess_xx = torch.zeros_like(autohess[:n, :n].real)
            numhess_xy = torch.zeros_like(autohess[:n, :n].real)
            numhess_yx = torch.zeros_like(autohess[:n, :n].real)
            numhess_yy = torch.zeros_like(autohess[:n, :n].real)
        else:
            numhess = torch.zeros_like(autohess)
        
        def evaluate_func_at_point(x):
            self.set_flat_params(flat_params=x, name_list=weight_names)
            return self.loss_function_val(signal_batch=signal_batch)

        delta_row, delta_col = torch.zeros_like(flat_params), torch.zeros_like(flat_params)
        for i in range(flat_params.numel()):
            delta_row[i - 1] = 0.
            delta_row[i] = 1.
            for j in range(flat_params.numel()):
                delta_col[j - 1] = 0.
                delta_col[j] = 1.
                if complex_derivative:
                    if order == 1:
                        numhess_xx[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + eps * delta_row) -
                            evaluate_func_at_point(flat_params + eps * delta_col) + func_val) / (eps ** 2)
                        numhess_yy[i, j] = (evaluate_func_at_point(flat_params + 1.j * eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + 1.j * eps * delta_row) -
                            evaluate_func_at_point(flat_params + 1.j * eps * delta_col) + func_val) / (eps ** 2)
                        numhess_xy[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + 1.j * delta_col)) -
                            evaluate_func_at_point(flat_params + eps * delta_row) -
                            evaluate_func_at_point(flat_params + 1.j * eps * delta_col) + func_val) / (eps ** 2)
                        numhess_yx[i, j] = (evaluate_func_at_point(flat_params + eps * (1.j * delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + 1.j * eps * delta_row) -
                            evaluate_func_at_point(flat_params + eps * delta_col) + func_val) / (eps ** 2)
                    else:
                        numhess_xx[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_row - delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_col - delta_row)) +
                            evaluate_func_at_point(flat_params - eps * (delta_row + delta_col))) / (4 * eps * eps)
                        numhess_yy[i, j] = (evaluate_func_at_point(flat_params + 1.j * eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + 1.j * eps * (delta_row - delta_col)) -
                            evaluate_func_at_point(flat_params + 1.j * eps * (delta_col - delta_row)) +
                            evaluate_func_at_point(flat_params - 1.j * eps * (delta_row + delta_col))) / (4 * eps * eps)
                        numhess_xy[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + 1.j * delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_row - 1.j * delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (1.j * delta_col - delta_row)) +
                            evaluate_func_at_point(flat_params - eps * (delta_row + 1.j * delta_col))) / (4 * eps * eps)
                        numhess_yx[i, j] = (evaluate_func_at_point(flat_params + eps * (1.j * delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (1.j * delta_row - delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_col - 1.j * delta_row)) +
                            evaluate_func_at_point(flat_params - eps * (1.j * delta_row + delta_col))) / (4 * eps * eps)
                else:
                    if order == 1:
                        numhess[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + eps * delta_row) -
                            evaluate_func_at_point(flat_params + eps * delta_col) + func_val) / (eps ** 2)
                    else:
                        numhess[i, j] = (evaluate_func_at_point(flat_params + eps * (delta_row + delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_row - delta_col)) -
                            evaluate_func_at_point(flat_params + eps * (delta_col - delta_row)) +
                            evaluate_func_at_point(flat_params - eps * (delta_row + delta_col))) / (4 * eps * eps)

        if complex_derivative:
            if return_full_wirtinger_derivative:
                H_z_z = (numhess_xx - numhess_yy - 1.j * (numhess_yx + numhess_xy)) / 4.
                H_z_z_conj = (numhess_xx + numhess_yy + 1.j * (numhess_xy - numhess_yx)) / 4.
                H_z_conj_z = (numhess_xx + numhess_yy + 1.j * (numhess_yx - numhess_xy)) / 4.
                H_z_conj_z_conj = (numhess_xx - numhess_yy + 1.j * (numhess_yx + numhess_xy)) / 4.
                numhess = torch.cat((torch.cat((H_z_z, H_z_z_conj), dim=1),
                    torch.cat((H_z_conj_z, H_z_conj_z_conj), dim=1)), dim=0)
            else:
                numhess = (numhess_xx + numhess_yy + 1.j * (numhess_yx - numhess_xy)) / 4.

        self.set_flat_params(flat_params=flat_params, name_list=weight_names)
        return torch.linalg.norm(autohess - numhess, ord=float("inf")).item()
