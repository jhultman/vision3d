import torch
from torch import nn


class ProposalLayer(nn.Module):
    """
    Use BEV feature map to generate 3D box proposals.
    TODO: Add orthogonal anchors.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.build_heads(cfg)
        self.cfg = cfg

    def build_heads(self, cfg):
        """Heads for box regression and classification."""
        self.cls = nn.Conv2d(cfg.PROPOSAL.C_IN, cfg.NUM_CLASSES, 1)
        self.reg = nn.Conv2d(cfg.PROPOSAL.C_IN, (cfg.NUM_CLASSES - 1) * cfg.BOX_DOF, 1)

    def inference(self, features):
        raise NotImplementedError

    def reshape_boxes(self, reg_map):
        N, C, H, W = reg_map.shape
        reg_map = reg_map.view(N, -1, self.cfg.BOX_DOF, H, W)
        return reg_map

    def forward(self, feature_map):
        cls_map = self.cls(feature_map)
        reg_map = self.reg(feature_map)
        reg_map = self.reshape_boxes(reg_map)
        return cls_map, reg_map
