"""
Modified SpMiddleFHD (see github.com/traveller59/second.pytorch).
"""

import itertools
import torch
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
    """Computes mean of non-zero points within voxel."""

    def __init__(self):
        super(VoxelFeatureExtractor, self).__init__()

    def forward(self, feature, occupancy):
        """
        :feature FloatTensor of shape (N, K, C)
        :return FloatTensor of shape (N, C)
        """
        denominator = occupancy.type_as(feature).view(-1, 1)
        feature = (feature.sum(1) / denominator).contiguous()
        return feature


class SparseCNN(nn.Module):
    """
    Returns feature volumes strided 1x, 2x, 4x, 8x, 8x.
        block_1: [ 4, 1600, 1280, 41] -> [32, 800, 640, 21]
        block_2: [32,  800,  640, 21] -> [64, 400, 320, 11]
        block_3: [64,  400,  320, 11] -> [64, 200, 160,  5]
        block_4: [64,  400,  320,  5] -> [64, 200, 160,  2]
    """

    def __init__(self, grid_shape, cfg):
        """:grid_shape voxel grid dimensions in ZYX order."""
        super(SparseCNN, self).__init__()
        self.grid_shape = grid_shape
        self.cfg = cfg
        self.base_voxel_size = torch.cuda.FloatTensor(cfg.voxel_size)
        self.voxel_offset = torch.cuda.FloatTensor(cfg.grid_bounds[:3])

        self.block1 = spconv.SparseSequential(
            make_subm_layer(cfg.cnn_C_in, 16, 3, indice_key="subm0", bias=False),
            make_subm_layer(16, 16, 3, indice_key="subm0", bias=False),
            make_sparse_conv_layer(16, 32, 3, 2, padding=1, bias=False),
        )
        self.block2 = spconv.SparseSequential(
            make_subm_layer(32, 32, 3, indice_key="subm1", bias=False),
            make_subm_layer(32, 32, 3, indice_key="subm1", bias=False),
            make_sparse_conv_layer(32, 64, 3, 2, padding=1, bias=False),
        )
        self.block3 = spconv.SparseSequential(
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_subm_layer(64, 64, 3, indice_key="subm2", bias=False),
            make_sparse_conv_layer(64, 64, 3, 2, padding=[0, 1, 1], bias=False),
        )
        self.block4 = spconv.SparseSequential(
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

    def to_global(self, stride, volume):
        """
        Convert integer voxel indices to metric coordinates.
        Indices are reversed ijk -> kji to maintain correspondence with xyz.
        :voxel_size length-3 tensor describing size of atomic voxel, accounting for stride.
        :voxel_offset length-3 tensor describing coordinate offset of voxel grid.
        """
        feature = volume.features
        index = torch.flip(volume.indices, (1,))
        voxel_size = self.base_voxel_size * stride
        xyz = index[..., 0:3].float() * voxel_size
        xyz = (xyz + self.voxel_offset)
        return xyz, feature

    def forward(self, features, coordinates, batch_size):
        x0 = spconv.SparseConvTensor(
            features, coordinates.int(), self.grid_shape, batch_size
        )
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        args = zip(self.cfg.strides, (x0, x1, x2, x3))
        x = list(itertools.starmap(self.to_global, args))
        return x, x4
