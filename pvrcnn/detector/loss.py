import torch
from torch import nn
import torch.nn.functional as F


class ProposalLoss(nn.Module):
    """
    TODO: Focal classification loss, binned angle loss.
    """

    def __init__(self, cfg):
        super(ProposalLoss, self).__init__()
        self.cfg = cfg

    def masked_average(self, loss, mask):
        mask = mask.type_as(loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def reg_loss(self, targets_reg, pred_reg, mask):
        """
        Loss is applied at all positive sites, averaged
        over the number of such sites.
        """
        G_xyz, G_wlh, G_yaw = targets_reg.split([3, 3, 1], dim=-1)
        P_xyz, P_wlh, P_yaw = pred_reg.split([3, 3, 1], dim=-1)
        loss_xyz = F.smooth_l1_loss(P_xyz, G_xyz, reduction='none')
        loss_wlh = F.smooth_l1_loss(P_wlh, G_wlh, reduction='none')
        loss_yaw = F.smooth_l1_loss(P_yaw, G_yaw, reduction='none')
        loss = self.masked_average(loss_xyz + loss_wlh + loss_yaw, mask)
        return loss

    def cls_loss(self, G_cls, P_cls, mask):
        """
        Assumes logit scores (not softmax rectified).
        Loss is applied at all non-ignore sites, averaged
        over the number of such sites.
        """
        _, G_cls = G_cls.max(dim=-1)
        P_cls = P_cls.transpose(1, 2)
        loss = F.cross_entropy(P_cls, G_cls, reduction='none')
        loss = self.masked_average(loss, mask)
        return loss

    def forward(self, item):
        """Proposal loss same as in Sparse-to-Dense."""
        keys = ['prop_targets_cls', 'prop_targets_reg', 'proposal_scores', 'proposal_boxes']
        targets_cls, targets_reg, pred_cls, pred_reg = map(item.__getitem__, keys)
        G_cls, mask_cls = targets_cls.split([self.cfg.NUM_CLASSES, 1], dim=-1)
        cls_loss = self.cls_loss(G_cls, pred_cls, mask_cls.squeeze(-1))
        reg_loss = self.reg_loss(targets_reg, pred_reg, 1 - G_cls[..., -1:, None])
        loss = cls_loss + self.cfg.TRAIN.LAMBDA * reg_loss
        return loss
