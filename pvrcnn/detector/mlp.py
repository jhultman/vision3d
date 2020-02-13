import torch
from torch import nn


class MLP(nn.Sequential):

    def __init__(self, channels, bias=False, bn=False, relu=True):
        super(MLP, self).__init__()
        bias, bn, relu = self.maybe_repeat(bias, bn, relu, len(channels))
        for i in range(len(channels) - 1):
            self.add_module(f'linear_{i}', nn.Linear(channels[i], channels[i+1], bias=bias[i]))
            if bn[i]:
                self.add_module(f'batchnorm_{i}', nn.BatchNorm1d(channels[i+1]))
            if relu[i]:
                self.add_module(f'relu_{i}', nn.ReLU(inplace=True))

    def maybe_repeat(self, bias, bn, relu, numel):
        if not isinstance(bias, (tuple, list)):
            bias = [bias] * (numel - 1)
        if not isinstance(bn, (tuple, list)):
            bn = [bn] * (numel - 1)
        if not isinstance(relu, (tuple, list)):
            relu = [relu] * (numel - 1)
        return bias, bn, relu
