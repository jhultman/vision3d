import torch
from torch import nn
from torch.nn import functional as F


class BEVFeatureGatherer(nn.Module):

    def __init__(self, cfg, voxel_offset, base_voxel_size):
        super(BEVFeatureGatherer, self).__init__()
        self.cfg = cfg
        self.pixel_offset = voxel_offset[:2]
        self.base_pixel_size = base_voxel_size[:2]

    def normalize_grid_sample_indices(self, indices, H, W):
        """F.grid_sample expects normalized indices on (-1, +1)."""
        image_dims = indices.new_tensor([H - 1, W - 1])
        indices = torch.min(torch.clamp(indices, min=0), image_dims)
        indices = 2 * (indices / (image_dims - 1)) - 1
        return indices

    def compute_bev_indices(self, keypoint_xyz, H, W):
        """Convert xyz coordinates to fractional BEV indices."""
        indices = keypoint_xyz[:, None, :, :2] - self.pixel_offset
        indices = indices / (self.base_pixel_size * self.cfg.STRIDES[-1])
        indices = self.normalize_grid_sample_indices(indices, H, W)
        return indices

    def forward(self, volume, keypoint_xyz):
        """
        Project 3D pixel grid to XY-plane and gather
        BEV features using bilinear interpolation.
        """
        volume = volume.dense()
        N, C, D, H, W = volume.shape
        volume = volume.view(N, C * D, H, W)
        indices = self.compute_bev_indices(keypoint_xyz, H, W)
        features = F.grid_sample(volume, indices).squeeze(2)
        return features
