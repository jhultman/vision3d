from copy import deepcopy
import torch
from torch import nn

from pointnet2.pointnet2_modules import PointnetSAModuleMSG

from .layers import MLP


class RoiGridPool(nn.Module):
    """
    Pools features from within proposals.
    TODO: I think must be misunderstanding dimensions claimed in paper.
        If sample 216 gridpoints in each proposal, and keypoint features
        are of dim 256, and gridpoint features are vectorized before linear layer,
        causes 216 * 256 * 256 parameters in reduction...
    TODO: Document input and output sizes.
    """

    def __init__(self, cfg):
        super(RoiGridPool, self).__init__()
        self.pnet = self.build_pointnet(cfg)
        self.reduction = MLP(cfg.GRIDPOOL.MLPS_REDUCTION)
        self.cfg = cfg

    def build_pointnet(self, cfg):
        """Copy channel list because PointNet modifies it in-place."""
        pnet = PointnetSAModuleMSG(
            npoint=-1, radii=cfg.GRIDPOOL.RADII_PN,
            nsamples=cfg.SAMPLES_PN,
            mlps=deepcopy(cfg.GRIDPOOL.MLPS_PN), use_xyz=True,
        )
        return pnet

    def rotate_z(self, points, theta):
        """
        Rotate points by theta around z-axis.
        :points (b, n, m, 3)
        :theta (b, n)
        :return (b, n, m, 3)
        """
        b, n, m, _ = points.shape
        theta = theta.unsqueeze(-1).expand(-1, -1, m)
        xy, z = torch.split(points, [2, 1], dim=-1)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack((c, -s, s, c), dim=-1).view(b, n, m, 2, 2)
        xy = torch.matmul(R, xy.unsqueeze(-1))
        xyz = torch.cat((xy.squeeze(-1), z), dim=-1)
        return xyz

    def sample_gridpoints(self, boxes):
        """
        Sample axis-aligned points, then rotate.
        :return (b, n, ng, 3)
        """
        b, n, _ = boxes.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = boxes[:, :, None, 3:6] * \
            (torch.rand((b, n, m, 3), device=boxes.device) - 0.5)
        gridpoints = boxes[:, :, None, 0:3] + \
            self.rotate_z(gridpoints, boxes[..., -1])
        return gridpoints

    def forward(self, proposals, keypoint_xyz, keypoint_features):
        b, n, _ = proposals.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = self.sample_gridpoints(proposals).view(b, -1, 3)
        features = self.pnet(keypoint_xyz, keypoint_features, gridpoints)[1]
        features = features.view(b, -1, n, m) \
            .permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        features = self.reduction(features)
        return features
