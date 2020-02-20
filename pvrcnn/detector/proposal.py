import torch
from torch import nn
import torch.nn.functional as F

from pvrcnn.ops import sigmoid_focal_loss, batched_nms_rotated


class ProposalLayer(nn.Module):
    """
    Use BEV feature map to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.build_heads(cfg)
        self.cfg = cfg

    def build_heads(self, cfg):
        """Heads for box regression and classification."""
        self.conv_cls = nn.Conv2d(
            cfg.PROPOSAL.C_IN, (cfg.NUM_CLASSES - 1) * cfg.NUM_YAW, 1)
        self.conv_reg = nn.Conv2d(
            cfg.PROPOSAL.C_IN, (cfg.NUM_CLASSES - 1) * cfg.NUM_YAW * cfg.BOX_DOF, 1)

    def inference(self, feature_map):
        """
        TODO: Sigmoid and topk proposal indexing.
        """
        cls_map, reg_map = self(feature_map)
        scores = cls_map.sigmoid()
        class_idx = torch.arange(self.cfg.NUM_CLASSES) - 1
        class_idx = class_idx[None, :, None, None, None].expand_as(scores)
        raise NotImplementedError

    def reshape_cls(self, cls_map):
        B, _, ny, nx = cls_map.shape
        shape = (B, self.cfg.NUM_CLASSES - 1, self.cfg.NUM_YAW, ny, nx)
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


class ProposalLoss(nn.Module):
    """
    Note: P_i and G_i refer to predicted and target quantities respectively.
    TODO: Binned angle loss.
    """

    def __init__(self, cfg):
        super(ProposalLoss, self).__init__()
        self.cfg = cfg

    def masked_average(self, loss, mask):
        mask = mask.type_as(loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def reg_loss(self, P_reg, G_reg, mask):
        """
        Loss is applied at all positive sites, averaged
        over the number of such sites.
        """
        P_xyz, P_wlh, P_yaw = P_reg.split([3, 3, 1], dim=-1)
        G_xyz, G_wlh, G_yaw = G_reg.split([3, 3, 1], dim=-1)
        loss_xyz = F.smooth_l1_loss(P_xyz, G_xyz, reduction='none')
        loss_wlh = F.smooth_l1_loss(P_wlh, G_wlh, reduction='none')
        loss_yaw = F.smooth_l1_loss(P_yaw, G_yaw, reduction='none')
        loss = self.masked_average(loss_xyz + loss_wlh + loss_yaw, mask)
        return loss

    def cls_loss(self, P_cls, G_cls, mask):
        """
        Assumes logit scores (not softmax rectified).
        Loss is applied at all non-ignore sites, averaged
        over the number of such sites.
        """
        loss = sigmoid_focal_loss(P_cls, G_cls, reduction='none')
        loss = self.masked_average(loss, mask)
        return loss

    def forward(self, item):
        G_cls, G_reg = map(item.get, ['proposal_targets_cls', 'proposal_targets_reg'])
        P_cls, P_reg = map(item.get, ['proposal_scores', 'proposal_boxes'])
        G_cls, mask_cls = G_cls.split([self.cfg.NUM_CLASSES - 1, 1], dim=1)
        mask_reg = G_cls[:, :-1, ..., None].sum(1, keepdim=True)
        cls_loss = self.cls_loss(P_cls, G_cls, mask_cls)
        reg_loss = self.reg_loss(P_reg, G_reg, mask_reg)
        loss = cls_loss + self.cfg.TRAIN.LAMBDA * reg_loss
        losses = dict(cls_loss=cls_loss, reg_loss=reg_loss, loss=loss)
        return losses
