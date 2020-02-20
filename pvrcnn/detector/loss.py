import torch
from torch import nn
import torch.nn.functional as F

from pvrcnn.thirdparty import sigmoid_focal_loss


class ProposalLoss(nn.Module):
    """TODO: Binned angle loss."""

    def __init__(self, cfg):
        super(ProposalLoss, self).__init__()
        self.cfg = cfg

    def masked_average(self, loss, mask):
        mask = mask.type_as(loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def reg_loss(self, pred_reg, targets_reg, mask):
        """
        Loss is applied at all positive sites, averaged
        over the number of such sites.
        """
        P_xyz, P_wlh, P_yaw = pred_reg.split([3, 3, 1], dim=-1)
        G_xyz, G_wlh, G_yaw = targets_reg.split([3, 3, 1], dim=-1)
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
        keys = ['proposal_targets_cls', 'proposal_targets_reg', 'proposal_scores', 'proposal_boxes']
        targets_cls, targets_reg, pred_cls, pred_reg = map(item.__getitem__, keys)
        G_cls, mask_cls = targets_cls.split([self.cfg.NUM_CLASSES - 1, 1], dim=1)
        mask_reg = G_cls[:, :-1, ..., None].sum(1, keepdim=True)
        cls_loss = self.cls_loss(pred_cls, G_cls, mask_cls)
        reg_loss = self.reg_loss(pred_reg, targets_reg, mask_reg)
        loss = cls_loss + self.cfg.TRAIN.LAMBDA * reg_loss
        losses = dict(cls_loss=cls_loss, reg_loss=reg_loss, loss=loss)
        return losses
