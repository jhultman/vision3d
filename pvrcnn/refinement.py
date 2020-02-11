import torch
from torch import nn

from pointnet2.pytorch_utils import FC

from pvrcnn.data_classes import Boxes3D


class RefinementLayer(nn.Module):
    """
    Use pooled features to refine proposals.
    """

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        """TODO: Ensure no batch norm or activation in regression."""
        mlp = nn.Sequential(
            FC(*cfg.REFINEMENT.MLPS[0:2]),
            FC(*cfg.REFINEMENT.MLPS[1:3]),
        )
        return mlp

    def apply_refinements(self, refinements, proposals):
        """TODO: Use proper box encoding."""
        return Boxes3D(refinements + proposals)

    def forward(self, proposals, features):
        refinements = self.mlp(features)
        predictions = self.apply_refinements(refinements, proposals.tensor)
        return predictions
