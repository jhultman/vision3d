import numpy as np
import torch
from torch import nn
from typing import List
from collections import defaultdict

import spconv


class Preprocessor(nn.Module):

    def __init__(self, cfg):
        super(Preprocessor, self).__init__()
        self.voxel_generator = self.build_voxel_generator(cfg)
        self.cfg = cfg

    def build_voxel_generator(self, cfg):
        """Voxel-grid is reversed XYZ -> ZYX and padded in Z-axis."""
        voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.VOXEL_SIZE,
            point_cloud_range=cfg.GRID_BOUNDS,
            max_voxels=cfg.MAX_VOXELS,
            max_num_points=cfg.MAX_OCCUPANCY,
        )
        return voxel_generator

    def generate_batch_voxels(self, points):
        """Voxelize points and tag with batch index."""
        features, coordinates, occupancy = [], [], []
        for i, p in enumerate(points):
            f, c, o = self.voxel_generator.generate(p)
            c = np.pad(c, ((0, 0), (1, 0)), constant_values=i)
            features += [f]; coordinates += [c]; occupancy += [o]
        return map(np.concatenate, (features, coordinates, occupancy))

    def pad_for_batch(self, points: List) -> np.ndarray:
        """
        Pad with subsampled points to form dense minibatch.
        :return np.ndarray of shape (B, N, C)
        """
        num_points = np.r_[[p.shape[0] for p in points]]
        pad = num_points.max() - num_points
        points_batch = []
        for points_i, pad_i in zip(points, pad):
            idx = np.random.choice(points_i.shape[0], pad_i)
            points_batch += [np.concatenate((points_i, points_i[idx]))]
        points = np.stack(points_batch, axis=0)
        return points

    def from_numpy(self, x):
        """Make cuda tensor."""
        return torch.from_numpy(x).cuda()

    def forward(self, item):
        """
        Compute sparse voxel grid.
        :points_in list of np.ndarrays of shape (Np, 4)
        :points_out FloatTensor of shape (Np, 4)
        :features FloatTensor of shape (Nv, 1)
        :coordinates IntTensor of shape (Nv, 4)
        """
        features, coordinates, occupancy = self.generate_batch_voxels(item['points'])
        points = self.pad_for_batch(item['points'])
        keys = ['points', 'features', 'coordinates', 'occupancy']
        vals = map(self.from_numpy, (points, features, coordinates, occupancy))
        item.update(dict(zip(keys, vals)))
        item['batch_size'] = len(points)
        return item


class TrainPreprocessor(Preprocessor):

    def __init__(self, cfg):
        super(TrainPreprocessor, self).__init__(cfg)

    def collate_mapping(self, key):
        torch_stack = ['proposal_targets_cls', 'proposal_targets_reg']
        identity = ['idx', 'points', 'boxes', 'class_idx']
        if key in torch_stack:
            return torch.stack
        return lambda x: x

    def collate(self, items):
        batch_item = defaultdict(list)
        for item in items:
            for key, val in item.items():
                if isinstance(val, torch.Tensor):
                    batch_item[key] += [val.cuda()]
                else:
                    batch_item[key] += [val]
        for key, val in batch_item.items():
            batch_item[key] = self.collate_mapping(key)(val)
        return self(dict(batch_item))
