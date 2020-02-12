import torch
from torch import nn

from pointnet2.pytorch_utils import FC

from pvrcnn.data_classes import Boxes3D


class ProposalLayer(nn.Module):
    """
    Use keypoint features to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        """TODO: Ensure no batch norm or activation in regression."""
        mlp = nn.Sequential(
            FC(*cfg.PROPOSAL.MLPS[0:2]),
            FC(*cfg.PROPOSAL.MLPS[1:3]),
        )
        return mlp

    def forward(self, points, features):
        features = features.permute(0, 2, 1)
        proposals = Boxes3D(self.mlp(features))
        _, indices = torch.topk(proposals.score, k=self.cfg.PROPOSAL.TOPK, dim=1)
        indices = indices.expand(-1, -1, proposals.tensor.shape[-1])
        proposals = Boxes3D(proposals.tensor.gather(1, indices))
        return proposals
