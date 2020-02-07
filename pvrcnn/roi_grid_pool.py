import torch
from torch import nn


class RoiGridPool(nn.Module):

    def __init__(self, cfg):
        super(RoiGridPool, self).__init__()

    def rotate_grid_points(self, grid_points, theta):
        """
        Grid points were chosen according to axis-aligned 3D
        bounding boxes. Here we rotate them using the proposal yaw angle.
        """
        pass

    def sample_grid_points(self, proposals):
        pass

    def forward(self, keypoint_xyz, keypoint_features, proposals):
        pass
