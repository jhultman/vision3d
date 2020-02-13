import torch
from torch import nn

from .mlp import MLP


class ProposalLayer(nn.Module):
    """
    Use keypoint features to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        mlp = MLP(cfg.PROPOSAL.MLPS, bias=True, bn=False, relu=[True, False])
        return mlp

    def forward(self, points, features):
        features = features.permute(0, 2, 1)
        proposals = self.mlp(features)
        _, indices = torch.topk(proposals[..., -1:], k=self.cfg.PROPOSAL.TOPK, dim=1)
        indices = indices.expand(-1, -1, proposals.shape[-1])
        proposals = proposals.gather(1, indices)
        return proposals[..., :-1], proposals[..., -1]
