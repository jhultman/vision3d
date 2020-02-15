import torch
from torch import nn
from functools import partial


class MLP(nn.Sequential):

    def __init__(self, channels, bias=False, bn=False, relu=True):
        super(MLP, self).__init__()
        bias, bn, relu = map(partial(self._repeat, n=len(channels)), (bias, bn, relu))
        for i in range(len(channels) - 1):
            self.add_module(f'linear_{i}', nn.Linear(channels[i], channels[i+1], bias=bias[i]))
            if bn[i]:
                self.add_module(f'batchnorm_{i}', nn.BatchNorm1d(channels[i+1]))
            if relu[i]:
                self.add_module(f'relu_{i}', nn.ReLU(inplace=True))

    def _repeat(self, module, n):
        if not isinstance(module, (tuple, list)):
            module = [module] * (n - 1)
        return module
