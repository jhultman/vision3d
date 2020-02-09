import numpy as np
import torch
from torch import nn

import spconv


class CNN_3D(nn.Module):
    """
    Simple placeholder sparse 3D CNN with four blocks:

        block_0: [12, 1600, 1280, 41] -> [16, 1600, 1280, 41]
        block_1: [16, 1600, 1280, 41] -> [16, 800, 640, 21]
        block_2: [16, 800, 640, 21]   -> [32, 400, 320, 11]
        block_3: [32, 400, 320, 11]   -> [64, 200, 160, 6]

    Input points within voxels are concatenated along channels.
    Returns feature volumes strided 1x, 2x, 4x, 8x.
    """

    def __init__(self, grid_shape, cfg):
        """
        :grid_shape voxel grid dimensions in ZYX order.
        """
        super(CNN_3D, self).__init__()
        C = cfg.C_in * cfg.max_num_points
        self.blocks = spconv.SparseSequential(
            spconv.SparseConv3d(C, 16, 3, 1, padding=0, bias=False),
            spconv.SparseConv3d(16, 16, 3, 2, padding=1, bias=False),
            spconv.SparseConv3d(16, 32, 3, 2, padding=1, bias=False),
            spconv.SparseConv3d(32, 64, 3, 2, padding=1, bias=False),
        )
        self.grid_shape = grid_shape
        self.base_voxel_size = torch.cuda.FloatTensor(cfg.voxel_size)
        self.voxel_offset = torch.cuda.FloatTensor(cfg.grid_bounds[:3])

    def to_global(self, stride, volume):
        """
        Convert integer voxel indices to metric coordinates.
        Indices are reversed ijk -> kji to maintain correspondence with xyz.

        voxel_size: length-3 tensor describing size of atomic voxel, accounting for stride.
        voxel_offset: length-3 tensor describing coordinate offset of voxel grid.
        """
        feature = volume.features
        index = torch.flip(volume.indices, (1,))
        voxel_size = self.base_voxel_size * stride
        xyz = index[..., 0:3].float() * voxel_size
        xyz = (xyz + self.voxel_offset)
        return xyz, feature

    def forward(self, features, coordinates, batch_size):
        x0 = spconv.SparseConvTensor(
            features, coordinates.int(), self.grid_shape, batch_size,
        )
        x1 = self.blocks[0](x0)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.blocks[3](x3)
        x = [self.to_global(2 ** i, x) for i, x in enumerate([x1, x2, x3, x4])]
        return x, x4
