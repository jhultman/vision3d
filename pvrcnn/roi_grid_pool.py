import torch
from torch import nn


class RoiGridPool(nn.Module):

    def __init__(self, cfg):
        super(RoiGridPool, self).__init__()
        self.cfg = cfg

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

    def sample_gridpoints(self, proposals, m):
        """
        Generate gridpoints within object proposals.
        :return FloatTensor of shape (nb, ng, 3)
        """
        n, device = proposals.shape[0], proposals.device
        gridpoints = torch.rand((n, m, 3), device=device) * proposals.wlh[:, None,]
        gridpoints = self.rotate_z(gridpoints, proposals.yaw) + proposals.center[:, None]
        return gridpoints

    def forward(self, proposals, keypoints):
        gridpoints = self.sample_gridpoints(proposals, self.cfg.n_gridpoints)
        return gridpoints
