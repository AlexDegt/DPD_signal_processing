import torch
import torch.nn as nn
from .complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexBatchNorm2d
from .batchnorm import Identity #ComplexBatchNorm1d
from typing import Union, List
from .activation import configure_activates
from collections import OrderedDict
import numpy as np

ListOfStr = List[str]
ListOfInt = List[int]
OptionalStr = Union[str, None]
OptionalList = Union[list, None]

class ComplexCNN(nn.Module):
    """
        Class of complex multi-layer convolutional neural-network (CNN).
    """
    def __init__(self, in_channels: int, out_channels: ListOfInt=[1, 1], kernel_size: ListOfInt=[5, 5], p_drop: OptionalList=None, activate: ListOfStr=['sigmoid', 'sigmoid'],
        batch_norm_mode: str='nothing', bias: bool=True, device: OptionalStr=None, dtype: torch.dtype=torch.complex128):
        """
        Constructor of the ComplexCNN class. 
        ComplexCNN consists of sequential layers of filters:
            input filter (4xK), filter (KxK), ..., filter (KxK), output filter (1xK).
        It means that input filter has 4 input channels and K output. The same for other layer.

        Args:
            in_channels (int): The number of CNN complex input channels. 
            out_channels (list of int): The number of CNN filters complex output channels. 
                i-th element of list corresponds to the number of output channels in i-th convolutional layer. Defaults to [1, 1].
            p_drop (list, optional): List of droput parameters. Each elements corresponds to the probability of an 
                element to be zeroed at each convolutional layer. Number of list elements must match number of convolutional layers. 
                If p_drop == None, then dropout of each layer is set to 0. Default to "None".
            activate (list of str): List of activation function names which are used in each conv. layer.
                i-th element of list corresponds to i-th conv. layer. Default to ['sigmoid', 'sigmoid'].
            batch_norm_mode (str): Type of batch normalization which is used in CNN.
                'common' -- Usual batch normalization: calcualtes variance, bias of the input batch and tunes adaptive scale and shift.
                    Attention!!! 'common' mode requires debuging or manual implementation. 
                    # Currently it exploits ComplexBatchNorm2d from complexPyTorch library.
                'nothing' -- The is no batcnh normalization, bypass.
                Default to 'nothing'.
            bias (bool): Parameter shows whether to exploit bias in convolutional layers or not. Default to True.
            device (str, optional): Parameter shows which device to use for calculation on.
                'cpu', None -- CPU usage.
                'cuda' -- GPU usage.
            dtype (torch.complex64 or torch.complex128): Parameter type. Default to torch.complex128.
        """
        super().__init__()
        assert len(activate) == len(out_channels), \
            "Number of activation functions must be the same as number of convolutional layers"
        if p_drop is None:
            p_drop = [0 for i in range(len(out_channels))]
        assert len(p_drop) == len(out_channels), \
            "Number of dropout parameters must be the same as number of convolutional layers"
        self.activate = activate
        # FC-layers initialization
        self.num_layers = len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_drop = p_drop
        # List of convolutional layers
        self.cnn = nn.Sequential()
        for layer_i in range(self.num_layers):

            dropout = nn.Dropout(self.p_drop[layer_i])

            conv_layer = nn.Conv1d(self.in_channels, self.out_channels[layer_i], self.kernel_size[layer_i],
                                 padding='valid', device=device, dtype=dtype, bias=bias)

            if batch_norm_mode == 'common':
                # Embedded nn.BatchNorm1d doesn`t work with complex data.
                # ComplexBatchNorm1d from complexPyTorch library works incorrectly.
                # Thus ComplexBatchNorm2d may be could be used, but requires tests.
                # Another way: manual implementation.
                fc_batchnorm = (ComplexBatchNorm2d(self.out_channels[layer_i], affine=True, eps=1e-05))
            if batch_norm_mode == 'nothing':
                fc_batchnorm = Identity()

            activation = configure_activates(activate[layer_i], channel_num=self.out_channels[layer_i], 
                                             dtype=dtype)

            self.cnn.append(nn.Sequential(OrderedDict([
                (f'dropout', dropout),
                (f'conv_layer', conv_layer),
                (f'fc_batchnorm', fc_batchnorm),
                (f'activation', activation)            
            ])))
            self.in_channels = self.out_channels[layer_i]

    def forward(self, x):
        # Conv1d-layers
        x_curr = x
        for layer_i in range(self.num_layers):
            x_curr = self.cnn[layer_i][0](x_curr)
            x_curr = self.cnn[layer_i][1](x_curr)
            # Dimension expansion is used for ComplexBatchNorm2d.
            x_curr = x_curr[..., None, :]
            x_curr = self.cnn[layer_i][2](x_curr)
            x_curr = x_curr[..., 0, :]
            x_curr = self.cnn[layer_i][3](x_curr)
        return x_curr

class RealCNN(nn.Module):
    """
        Class of real multi-layer convolutional neural-network (CNN).
    """
    def __init__(self, in_channels: int, out_channels: ListOfInt=[1, 1], kernel_size: ListOfInt=[5, 5], p_drop: OptionalList=None, activate: ListOfStr=['sigmoid', 'sigmoid'],
        batch_norm_mode: str='nothing', bias: bool=True, device: OptionalStr=None, dtype: torch.dtype=torch.float64):
        """
        Constructor of the RealCNN class. 
        RealCNN consists of sequential layers of filters:
            input filter (6xK), filter (KxK), ..., filter (KxK), output filter (2xK).
        It means that input filter has 6 input channels and K output. The same for other layer.

        Args:
            in_channels (int): The number of CNN complex input channels. 
            out_channels (list of int): The number of CNN filters real output channels. 
                i-th element of list corresponds to the number of output channels in i-th convolutional layer. Defaults to [1, 1].
            p_drop (list, optional): List of droput parameters. Each elements corresponds to the probability of an 
                element to be zeroed at each convolutional layer. Number of list elements must match number of convolutional layers. 
                If p_drop == None, then dropout of each layer is set to 0. Default to "None".
            activate (list of str): List of activation function names which are used in each conv. layer.
                i-th element of list corresponds to i-th conv. layer. Default to ['sigmoid', 'sigmoid'].
            batch_norm_mode (str): Type of batch normalization which is used in CNN.
                'common' -- Usual batch normalization: calcualtes variance, bias of the input batch and tunes adaptive scale and shift.
                'nothing' -- The is no batcnh normalization, bypass.
                Default to 'nothing'.
            bias (bool): Parameter shows whether to exploit bias in convolutional layers or not. Default to True.
            device (str, optional): Parameter shows which device to use for calculation on.
                'cpu', None -- CPU usage.
                'cuda' -- GPU usage.
            dtype (torch.float32 or torch.float64): Parameter type. Default to torch.float64.
        """
        super().__init__()
        assert len(activate) == len(out_channels), \
            "Number of activation functions must be the same as number of convolutional layers"
        if p_drop is None:
            p_drop = [0 for i in range(len(out_channels))]
        assert len(p_drop) == len(out_channels), \
            "Number of dropout parameters must be the same as number of convolutional layers"
        self.activate = activate
        # FC-layers initialization
        self.num_layers = len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_drop = p_drop
        # List of convolutional layers
        self.cnn = nn.Sequential()
        for layer_i in range(self.num_layers):

            dropout = nn.Dropout(self.p_drop[layer_i])

            conv_layer = nn.Conv1d(self.in_channels, self.out_channels[layer_i], self.kernel_size[layer_i],
                                 padding='valid', device=device, dtype=dtype, bias=bias)

            if batch_norm_mode == 'common':
                fc_batchnorm = nn.BatchNorm1d(self.out_channels[layer_i], affine=True, eps=1e-05)
            if batch_norm_mode == 'nothing':
                fc_batchnorm = Identity()

            activation = configure_activates(activate[layer_i], channel_num=self.out_channels[layer_i], 
                                             dtype=dtype)

            self.cnn.append(nn.Sequential(OrderedDict([
                (f'dropout', dropout),
                (f'conv_layer', conv_layer),
                (f'fc_batchnorm', fc_batchnorm),
                (f'activation', activation)            
            ])))
            self.in_channels = self.out_channels[layer_i]

    def forward(self, x):
        # Conv1d-layers
        x_curr = x
        for layer_i in range(self.num_layers):
            for sub_layer in self.cnn[layer_i]:
                x_curr = sub_layer(x_curr)
        return x_curr