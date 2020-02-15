from copy import deepcopy
import torch
from torch import nn

from pointnet2.pointnet2_modules import PointnetSAModuleMSG

from .mlp import MLP


class RoiGridPool(nn.Module):

    def __init__(self, cfg):
        super(RoiGridPool, self).__init__()
        self.pnet = self.build_pointnet(cfg)
        self.reduction = self.build_reduction(cfg)
        self.cfg = cfg

    def build_pointnet(self, cfg):
        """Copy channel list because PointNet modifies it in-place."""
        pnet = PointnetSAModuleMSG(
            npoint=-1, radii=cfg.GRIDPOOL.RADII_PN,
            nsamples=cfg.SAMPLES_PN,
            mlps=deepcopy(cfg.GRIDPOOL.MLPS_PN), use_xyz=True,
        )
        return pnet

    def build_reduction(self, cfg):
        reduction = MLP(cfg.GRIDPOOL.MLPS_REDUCTION)
        return reduction

    def rotate_z(self, points, theta):
        """
        Rotate points by theta around z-axis.
        :points FloatTensor of shape (b, n, m, 3)
        :theta FloatTensor of shape (b, n)
        :return FloatTensor of shape (b, n, m, 3)
        """
        b, n, m, _ = points.shape
        theta = theta.unsqueeze(-1).expand(-1, -1, m)
        xy, z = torch.split(points, [2, 1], dim=-1)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack((c, -s, s, c), dim=-1).view(b, n, m, 2, 2)
        xy = torch.matmul(R, xy.unsqueeze(-1))
        xyz = torch.cat((xy.squeeze(-1), z), dim=-1)
        return xyz

    def sample_gridpoints(self, proposals):
        """
        Generate gridpoints within axis-aligned
        object proposals then rotate about z-axis.
        :return FloatTensor of shape (nb, ng, 3)
        """
        b, n, _ = proposals.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = torch.rand((b, n, m, 3), device=proposals.device) * \
            proposals[:, :, None, 3:6]
        gridpoints = self.rotate_z(gridpoints, proposals[..., -1]) + \
            proposals[:, :, None, 0:3]
        return gridpoints

    def forward(self, proposals, keypoint_xyz, keypoint_features):
        """
        Gather features from within proposals.
        TODO: Ensure gridpoint features are reduced correctly.
        """
        b, n, _ = proposals.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = self.sample_gridpoints(proposals)
        gridpoints = gridpoints.view(b, -1, 3)
        pooled_features = self.pnet(keypoint_xyz, keypoint_features, gridpoints)[1]
        pooled_features = pooled_features.view(b, -1, n, m) \
            .permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        pooled_features = self.reduction(pooled_features)
        return pooled_features
