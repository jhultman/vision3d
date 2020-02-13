import torch
from torch import nn

from .mlp import MLP


class RefinementLayer(nn.Module):
    """Use pooled features to refine proposals."""

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        mlp = MLP(cfg.REFINEMENT.MLPS, bias=True, bn=False, relu=[True, False])
        return mlp

    def apply_refinements(self, refinements, proposals):
        """TODO: Use proper box encoding."""
        return refinements + proposals

    def forward(self, proposals, features):
        refinements = self.mlp(features)
        predictions = self.apply_refinements(refinements[..., :-1], proposals)
        return predictions, predictions[..., -1]
