"""
Code borrowed from https://github.com/traveller59/second.pytorch.
"""

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import spconv


def build_batchnorm(C_in):
    layer = nn.BatchNorm1d(C_in, eps=1e-3, momentum=0.01)
    for param in layer.parameters():
        param.requires_grad = True
    return layer


def make_subm_layer(C_in, C_out, *args, **kwargs):
    layer = spconv.SparseSequential(
        spconv.SubMConv3d(C_in, C_out, 3, *args, **kwargs),
        build_batchnorm(C_out),
        nn.ReLU(),
    )
    return layer


def make_sparse_conv_layer(C_in, C_out, *args, **kwargs):
    layer = spconv.SparseSequential(
        spconv.SparseConv3d(C_in, C_out, *args, **kwargs),
        build_batchnorm(C_out),
        nn.ReLU(),
    )
    return layer


class VoxelFeatureExtractor(nn.Module):

    def __init__(self, C_in):
        super(VoxelFeatureExtractor, self).__init__()
        self.C_in = C_in

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, : self.C_in].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class SparseCNN(nn.Module):
    """
    Based on SECOND SpMiddleFHD.

        [1600, 1200, 41] -> [800, 600, 21]
        [800, 600, 21]   -> [400, 300, 11]
        [400, 300, 11]   -> [200, 150, 5]
        [200, 150, 5]    -> [200, 150, 2]
    """

    def __init__(self, grid_shape, C_in):
        super(SparseCNN, self).__init__()
        self.grid_shape = grid_shape
        self.block0 = spconv.SparseSequential(
            make_subm_layer(C_in, 16, 3, indice_key="subm0", bias=False),
            make_subm_layer(16, 16, 3, indice_key="subm0", bias=False),
            make_sparse_conv_layer(16, 32, 3, 2, padding=1, bias=False),
        )
        self.block1 = spconv.SparseSequential(
            make_subm_layer(32, 32, 3, indice_key="subm1", bias=False),
            make_subm_layer(32, 32, 3, indice_key="subm1", bias=False),
            make_sparse_conv_layer(32, 64, 3, 2, padding=1, bias=False),
        )
        self.block2 = spconv.SparseSequential(
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_sparse_conv_layer(64, 64, 3, 2, padding=[0, 1, 1], bias=False),
        )
        self.block3 = spconv.SparseSequential(
            make_subm_layer(64, 64, 3, indice_key="subm3", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm3", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm3", bias=False),
            make_sparse_conv_layer(64, 64, (3, 1, 1), (2, 1, 1), bias=False),
        )

    def maybe_bias_init(self, module, val):
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, val)

    def kaiming_init(self, module):
        nn.init.kaiming_normal_(
            module.weight, a=0, mode='fan_out', nonlinearity='relu')
        self.maybe_bias_init(module, 0)

    def batchnorm_init(self, module):
        nn.init.constant_(module.weight, 1)
        self.maybe_bias_init(module, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                self.batchnorm_init(m)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(
            voxel_features, coors, self.grid_shape, batch_size)
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = [x0, x1, x2, x3]
        return x
