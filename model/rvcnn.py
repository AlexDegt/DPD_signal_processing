import torch, sys
from .layers import FEAT_EXTR, RealCNN
from itertools import chain

class RVCNN(torch.nn.Module):
    '''
        RVCNN - Real-Valued Convolutional NN.
        Takes pure signal for each of N channels x_{0, n}, ..., x_{N-1, n} as an input and 
        creates input features: Re(x_{0, n}), Im(x_{0, n}), |x_{0, n}|, ..., Re(x_{N-1, n}), Im(x_{N-1, n}), |x_{N-1, n}|. 
        Thus there're 3N input channels. Output channel numbers are regulated by the list out_channels.
        Last layer output channels number equal 2, which correspond to Re(x_last_layer) and Im(x_last_layer) part of 
        pre-distorted signal. Output of the RVCNN is Re(x_last_layer) + 1j * Im(x_last_layer)
    '''
    def __init__(self, delays=[[0]], out_channels=[1, 1], kernel_size=[5, 5],
        p_drop=None, activate=['sigmoid', 'sigmoid'], batch_norm_mode='nothing', features=['real', 'imag', 'abs'],
        bias=True, device=None, dtype=torch.float64):
        super().__init__()
        self._dtype = dtype
        self._device = device

        self._delay_num = len(list(chain(*delays)))
        self.parallel_num = len(delays)
        self.input_channel_num = 2

        # Feature extractor module is used to create 6 input channels from 2: 
        # Re(x_{A, n}), Im(x_{A, n}), Re(x_{B, n}), Im(x_{B, n}), |x_{A, n}|, |x_{B, n}|.
        self.feature_extract = FEAT_EXTR(features=features, device=device, dtype=dtype)
        
        # Complex convolutional NN.
        self.nonlin = RealCNN(in_channels=len(features)*self.input_channel_num, 
                              out_channels=out_channels, kernel_size=kernel_size, p_drop=p_drop, 
                              activate=activate, batch_norm_mode=batch_norm_mode, 
                              bias=bias, device=device, dtype=dtype)
        
    def forward(self, x):
        x_curr = self.feature_extract(x)
        x_curr = self.nonlin(x_curr)
        output = x_curr[:, :1, :] + 1j * x_curr[:, 1:2, :]
        return output