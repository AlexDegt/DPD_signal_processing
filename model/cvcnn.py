import torch
from .layers import FEAT_EXTR, ComplexCNN
from itertools import chain

class CVCNN(torch.nn.Module):
    '''
        CVCNN - Complex-Valued Convolutional NN.
        Takes pure signal both channels x_{A, n}, x_{B, n} as an input and 
        creates input features: x_{A, n}, x_{B, n}, |x_{A, n}|, |x_{B, n}|. Thus there're 4 input channels.
        Output channel numbers are regulated by the list out_channels.
        Last layer output channels number equal 1, which corresponds to pre-distorted signal.
    '''
    def __init__(self, delays=[[0]], out_channels=[1, 1], kernel_size=[5, 5],
        p_drop=None, activate=['sigmoid', 'sigmoid'], batch_norm_mode='nothing', features=['same', 'abs'],
        bias=True, device=None, dtype=torch.complex128):
        super().__init__()
        self._dtype = dtype
        self._device = device

        self._delay_num = len(list(chain(*delays)))
        self.parallel_num = len(delays)
        self.input_channel_num = 2

        # Feature extractor module is used to create 4 input channels from 2: x_{A, n}, x_{B, n}, |x_{A, n}|, |x_{B, n}|.
        self.feature_extract = FEAT_EXTR(features=features, device=device, dtype=dtype)
        
        # Complex convolutional NN.
        self.nonlin = ComplexCNN(in_channels=len(features)*self.input_channel_num, 
                              out_channels=out_channels, kernel_size=kernel_size, p_drop=p_drop, 
                              activate=activate, batch_norm_mode=batch_norm_mode, 
                              bias=bias, device=device, dtype=dtype)
        
    def forward(self, x):
        x_curr = self.feature_extract(x)
        output = self.nonlin(x_curr)
        return output