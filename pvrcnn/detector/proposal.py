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
        self.conv_cls = nn.Conv2d(
            cfg.PROPOSAL.C_IN, cfg.NUM_CLASSES * cfg.NUM_YAW, 1)
        self.conv_reg = nn.Conv2d(
            cfg.PROPOSAL.C_IN, (cfg.NUM_CLASSES - 1) * cfg.NUM_YAW * cfg.BOX_DOF, 1)

    def inference(self, features):
        """TODO: Topk proposal indexing."""
        raise NotImplementedError

    def reshape_cls(self, cls_map):
        B, _, ny, nx = cls_map.shape
        shape = (B, self.cfg.NUM_CLASSES, self.cfg.NUM_YAW, ny, nx)
        cls_map = cls_map.view(shape)
        return cls_map

    def reshape_reg(self, reg_map):
        B, _, ny, nx = reg_map.shape
        shape = (B, self.cfg.NUM_CLASSES - 1, self.cfg.BOX_DOF, -1, ny, nx)
        reg_map = reg_map.view(shape).permute(0, 1, 3, 4, 5, 2)
        return reg_map

    def forward(self, feature_map):
        cls_map = self.reshape_cls(self.conv_cls(feature_map))
        reg_map = self.reshape_reg(self.conv_reg(feature_map))
        return cls_map, reg_map
