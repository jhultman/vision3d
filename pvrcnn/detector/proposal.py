import torch
from torch import nn


class ProposalLayer(nn.Module):
    """
    Use BEV feature map to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.build_heads(cfg)
        self.cfg = cfg

    def build_heads(self, cfg):
        """
        Heads for box regression and classification.
        TODO: Add orthogonal anchors.
        """
        self.conv_cls = nn.Conv2d(cfg.PROPOSAL.C_IN, cfg.NUM_CLASSES, 1)
        self.conv_reg = nn.Conv2d(cfg.PROPOSAL.C_IN, cfg.BOX_DOF, 1)

    def inference(self, features):
        """TODO: Topk proposal indexing."""
        raise NotImplementedError

    def forward(self, feature_map):
        cls_map = self.conv_cls(feature_map)
        reg_map = self.conv_reg(feature_map)
        return cls_map, reg_map
