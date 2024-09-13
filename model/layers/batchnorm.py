import torch
import torch.nn as nn

class ScaleShift(nn.Module):
    def __init__(self, channel_num, dtype):
        super().__init__()
        self.channel_num = channel_num
        self.weight = torch.nn.Parameter(torch.ones(channel_num, dtype=dtype), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(channel_num, dtype=dtype), requires_grad=True)
    def forward(self, x):
        bias = self.bias.expand(x.size()[0], x.size()[2], self.channel_num)
        bias = torch.permute(bias, (0, 2, 1))
        weight = self.weight.expand(x.size()[0], x.size()[2], self.channel_num)
        weight = torch.permute(weight, (0, 2, 1))
        x *= weight
        x += bias
        return x
    
class ComplexBatchNorm1d(nn.Module):
    def __init__(self, channel_num, dtype):
        super().__init__()
        self.channel_num = channel_num
        self.weight = torch.nn.Parameter(torch.ones((1, channel_num, 1), dtype=dtype), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, channel_num, 1), dtype=dtype), requires_grad=True)
    def forward(self, x):
        # bias = self.bias.expand(x.size()[0], x.size()[2], self.channel_num)
        # bias = torch.permute(bias, (0, 2, 1))
        # weight = self.weight.expand(x.size()[0], x.size()[2], self.channel_num)
        # weight = torch.permute(weight, (0, 2, 1))
        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)
        mean_val = torch.mean(x, dim=(0, 2), keepdim=True)
        std_val = torch.sqrt(torch.var(x, dim=(0, 2), correction=0, keepdim=True) + 1e-8)
        mean_val = mean_val.detach()
        std_val = std_val.detach()
        # print(mean_val.shape, std_val.shape, x.shape)
        mean_val = mean_val.expand_as(x)
        std_val = std_val.expand_as(x)
        # print(mean_val.shape, std_val.shape, x.shape)
        # mean_val = torch.permute(mean_val, (1, 0)).expand_as(x)
        # std_val = torch.permute(std_val, (1, 0)).expand_as(x)
        # print(mean_val.shape, std_val.shape, x.shape)
        # print(std_val[])
        # sys.exit()
        x -= mean_val
        x *= weight
        x /= std_val
        x += bias
        return x

class Identity(nn.Module):
    '''
        Empty class that returns input tensor
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x