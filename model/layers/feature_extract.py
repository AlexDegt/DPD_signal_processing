import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SAME(nn.Module):
    def __init__(self):
        super(SAME, self).__init__()
    def forward(self, x):
        return x

class REAL(nn.Module):
    def __init__(self):
        super(REAL, self).__init__()
    def forward(self, x):
        return torch.real(x)
    
class IMAG(nn.Module):
    def __init__(self):
        super(IMAG, self).__init__()
    def forward(self, x):
        return torch.imag(x)
    
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()
    def forward(self, x):
        return torch.abs(x)
    
class ANGLE(nn.Module):
    def __init__(self):
        super(ANGLE, self).__init__()
    def forward(self, x):
        return torch.angle(x)

activat_funcs = {
    'same':   SAME(),
    'real':   REAL(),
    'imag':   IMAG(),
    'abs':    ABS(),
    'phase':  ANGLE(),
}

class FEAT_EXTR(nn.Module):
    def __init__(self, features, device=None, dtype=torch.complex128):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.features = features
        self.feature_num = len(features)

    def forward(self, x):
        input_num = x.shape[1]
        output = torch.zeros(x.shape[0], self.feature_num * input_num, x.shape[2], dtype=self.dtype, device=self.device)
        for j, feature in enumerate(self.features):
            output[..., j*input_num:(j+1)*input_num, :] = activat_funcs[feature](x)
        return output