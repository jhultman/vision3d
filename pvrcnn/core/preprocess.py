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
        voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.VOXEL_SIZE,
            point_cloud_range=cfg.GRID_BOUNDS,
            max_voxels=cfg.MAX_VOXELS,
            max_num_points=cfg.MAX_OCCUPANCY,
        )
        return voxel_generator

    def generate_batch_voxels(self, points):
        """Voxelize points and prefix coordinates with batch index."""
        features, coordinates, occupancy = [], [], []
        for i, p in enumerate(points):
            f, c, o = self.voxel_generator.generate(p)
            c = np.pad(c, ((0, 0), (1, 0)), constant_values=i)
            features += [f]; coordinates += [c]; occupancy += [o]
        return map(np.concatenate, (features, coordinates, occupancy))

    def pad_for_batch(self, points: List) -> np.ndarray:
        """Pad with subsampled points to form dense minibatch.
        :return np.ndarray of shape (B, N, C)"""
        num_points = np.r_[[p.shape[0] for p in points]]
        pad = num_points.max() - num_points
        points_batch = []
        for points_i, pad_i in zip(points, pad):
            idx = np.random.choice(points_i.shape[0], pad_i)
            points_batch += [np.concatenate((points_i, points_i[idx]))]
        points = np.stack(points_batch, axis=0)
        return points

    def forward(self, item):
        """
        Compute batch input from points.
        :points_in length B list of np.ndarrays of shape (Np, 4)
        :points_out FloatTensor of shape (B, Np, 4)
        :features FloatTensor of shape (B * Nv, 1)
        :coordinates IntTensor of shape (B * Nv, 4)
        :occupancy LongTensor of shape (B * Nv, 4)
        """
        features, coordinates, occupancy = self.generate_batch_voxels(item['points'])
        points = self.pad_for_batch(item['points'])
        keys = ['points', 'features', 'coordinates', 'occupancy', 'batch_size']
        vals = map(torch.from_numpy, (points, features, coordinates, occupancy))
        item.update(dict(zip(keys, list(vals) + [len(points)])))
        return item


class TrainPreprocessor(Preprocessor):

    def collate_mapping(self, key, val):
        if key in ['G_cls', 'G_reg', 'M_cls', 'M_reg']:
            return torch.stack(val)
        return val

    def collate(self, items):
        """Form batch item from list of items."""
        batch_item = defaultdict(list)
        for item in items:
            for key, val in item.items():
                batch_item[key] += [val]
        for key, val in batch_item.items():
            batch_item[key] = self.collate_mapping(key, val)
        return self(dict(batch_item))
