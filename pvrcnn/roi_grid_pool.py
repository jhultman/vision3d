import torch
from torch import nn

from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pytorch_utils import FC


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
            mlps=cfg.GRIDPOOL.MLPS_PN.copy(), use_xyz=True,
        )
        return pnet

    def build_reduction(self, cfg):
        reduction = nn.Sequential(
            FC(*cfg.GRIDPOOL.MLPS_REDUCTION[0:2]),
            FC(*cfg.GRIDPOOL.MLPS_REDUCTION[1:3]),
        )
        return reduction

    def rotate_z(self, points, theta):
        """
        Rotate points by theta around z-axis.
        :points FloatTensor of shape (N, M, 3)
        :theta FloatTensor of shape (N,)
        :return FloatTensor of shape (N, M, 3)
        """
        xy, z = torch.split(points, [2, 1], dim=-1)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack((c, -s, s, c), dim=-1).view(-1, 2, 2)
        xyz = torch.cat((torch.einsum('ijk,imk->imj', R, xy), z), dim=-1)
        return xyz

    def sample_gridpoints(self, proposals):
        """
        Generate gridpoints within object proposals.
        :return FloatTensor of shape (nb, ng, 3)
        """
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        n, device = proposals.wlh.shape[0], proposals.wlh.device
        gridpoints = torch.rand((n, m, 3), device=device) * proposals.wlh[:, None,]
        gridpoints = self.rotate_z(gridpoints, proposals.yaw) + proposals.center[:, None]
        return gridpoints

    def forward(self, proposals, keypoints_xyz, keypoints_features):
        """
        Gather features from within proposals.
        TODO: Ensure gridpoint features are reduced correctly.
        """
        gridpoints = self.sample_gridpoints(proposals)
        gridpoints = gridpoints.view(1, -1, 3)
        pooled_features = self.pnet(keypoints_xyz, keypoints_features, gridpoints)[1]
        n = proposals.wlh.shape[0]
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        pooled_features = pooled_features.view(1, -1, n, m) \
            .permute(0, 3, 1, 2).contiguous().view(1, n, -1)
        pooled_features = self.reduction(pooled_features)
        return pooled_features
