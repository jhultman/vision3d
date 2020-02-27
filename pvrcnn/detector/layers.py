import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


class VoxelFeatureExtractor(nn.Module):
    """Computes mean of non-zero points within voxel."""

    def forward(self, feature, occupancy):
        """
        :feature FloatTensor of shape (N, K, C)
        :return FloatTensor of shape (N, C)
        """
        denominator = occupancy.type_as(feature).view(-1, 1)
        feature = (feature.sum(1) / denominator).contiguous()
        return feature


class BEVFeatureGatherer(nn.Module):
    """Gather BEV features at keypoints using bilinear interpolation."""

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
        N, C, H, W = feature_map.shape
        indices = self.compute_bev_indices(keypoint_xyz, H, W)
        features = F.grid_sample(feature_map, indices, align_corners=True).squeeze(2)
        return features


class MLP(nn.Sequential):

    def __init__(self, channels, bias=False, bn=False, relu=True):
        super(MLP, self).__init__()
        bias, bn, relu = map(partial(self._repeat, n=len(channels)), (bias, bn, relu))
        for i in range(len(channels) - 1):
            self.add_module(f'linear_{i}', nn.Linear(channels[i], channels[i+1], bias=bias[i]))
            nn.init.normal_(self[-1].weight, std=0.01)
            if bias[i]:
                nn.init.constant_(self[-1].bias, 0)
            if bn[i]:
                self.add_module(f'batchnorm_{i}', nn.BatchNorm1d(channels[i+1]))
                nn.init.constant_(self[-1].weight, 1)
                nn.init.constant_(self[-1].bias, 0)
            if relu[i]:
                self.add_module(f'relu_{i}', nn.ReLU(inplace=True))

    def _repeat(self, module, n):
        if not isinstance(module, (tuple, list)):
            module = [module] * (n - 1)
        return module
