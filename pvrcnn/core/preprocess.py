import numpy as np
import torch
from torch import nn
from typing import List

import spconv
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

from .target_assigner import TargetAssigner


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
        self.grid_shape = np.r_[voxel_generator.grid_size[::-1]] + [1, 0, 0]
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

    def voxelize(self, points):
        """
        Compute sparse voxel grid.
        :points_in list of np.ndarrays of shape (Np, 4)
        :points_out FloatTensor of shape (Np, 4)
        :features FloatTensor of shape (Nv, 1)
        :coordinates IntTensor of shape (Nv, 4)
        """
        features, coordinates, occupancy = self.generate_batch_voxels(points)
        points = self.pad_for_batch(points)
        keys = ['points', 'features', 'coordinates', 'occupancy']
        vals = map(self.from_numpy, (points, features, coordinates, occupancy))
        input_dict = dict(zip(keys, vals))
        input_dict['batch_size'] = len(points)
        return input_dict

    def sample_keypoints(self, points):
        """
        Sample keypoints from raw pointcloud.
            - fps expects points shape (B, N, 3)
            - fps returns indices shape (B, K)
            - gather expects features shape (B, C, N)
        """
        points = points[..., :3].contiguous()
        indices = furthest_point_sample(points, self.cfg.NUM_KEYPOINTS)
        keypoints = gather_operation(points.transpose(1, 2).contiguous(), indices)
        keypoints = keypoints.transpose(1, 2).contiguous()
        return keypoints

    def forward(self, input_dict):
        input_dict.update(self.voxelize(input_dict['points']))
        input_dict['keypoints'] = self.sample_keypoints(input_dict['points'])
        return input_dict


class TrainPreprocessor(Preprocessor):

    def __init__(self, cfg):
        super(TrainPreprocessor, self).__init__(cfg)
        self.target_assigner = TargetAssigner(cfg)

    def forward(self, input_dict):
        input_dict.update(self.voxelize(input_dict['points']))
        input_dict['keypoints'] = self.sample_keypoints(input_dict['points'])
        input_dict = self.target_assigner(input_dict)
        return input_dict
