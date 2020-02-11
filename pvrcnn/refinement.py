import torch
from torch import nn

from pvrcnn.data_classes import Boxes3D


class RefinementLayer(nn.Module):
    """
    Use pooled features to refine proposals.
    """

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.cfg = cfg

    def forward(self, proposals, features):
        predictions = Boxes3D(20 * torch.rand((25, 7)).cuda())
        return predictions
