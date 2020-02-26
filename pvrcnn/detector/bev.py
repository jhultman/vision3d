import torch
from torch import nn
from torch.nn import functional as F


class BEVFeatureGatherer(nn.Module):
    """
    TODO: Does this class really need to live in its
        own file? Codebase is very fragmented.
    """

    def __init__(self, cfg, voxel_offset, base_voxel_size):
        super(BEVFeatureGatherer, self).__init__()
        self.cfg = cfg
        self.pixel_offset = voxel_offset[:2]
        self.base_pixel_size = base_voxel_size[:2]

    def normalize_indices(self, indices, H, W):
        """
        F.grid_sample expects normalized indices on (-1, +1).
        Note: We swap H and W because spconv transposes the feature map.
        """
        image_dims = indices.new_tensor([W - 1, H - 1])
        indices = torch.min(torch.clamp(indices, min=0), image_dims)
        indices = 2 * (indices / (image_dims - 1)) - 1
        return indices

    def compute_bev_indices(self, keypoint_xyz, H, W):
        """Convert xyz coordinates to fractional BEV indices."""
        indices = keypoint_xyz[:, None, :, :2] - self.pixel_offset
        indices = indices / (self.base_pixel_size * self.cfg.STRIDES[-1])
        indices = self.normalize_indices(indices, H, W).flip(3)
        return indices

    def forward(self, feature_map, keypoint_xyz):
        """
        Project 3D pixel grid to XY-plane and gather
        BEV features using bilinear interpolation.
        """
        N, C, H, W = feature_map.shape
        indices = self.compute_bev_indices(keypoint_xyz, H, W)
        features = F.grid_sample(feature_map, indices, align_corners=True).squeeze(2)
        return features
