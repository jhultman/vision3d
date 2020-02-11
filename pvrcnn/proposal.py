import torch
from torch import nn

from pvrcnn.data_classes import Boxes3D


class ProposalLayer(nn.Module):
    """
    Use keypoint features to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.cfg = cfg

    def forward(self, points, features):
        proposals = Boxes3D(20 * torch.rand((25, 7)).cuda())
        return proposals
