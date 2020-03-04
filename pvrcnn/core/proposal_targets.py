import torch
import math
from torch import nn

from pvrcnn.ops import box_iou_rotated, Matcher
from .anchor_generator import AnchorGenerator


class ProposalTargetAssigner(nn.Module):
    """
    Match ground truth boxes to anchors by IOU.
    TODO: Make this run faster if possible.
    """

    def __init__(self, cfg):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.anchors = AnchorGenerator(cfg).anchors.cuda()
        self.matchers = self.build_matchers(cfg)

    def build_matchers(self, cfg):
        matchers = []
        for anchor in cfg.ANCHORS:
            matchers += [Matcher(anchor['iou_thresh'],
                [0, -1, +1], cfg.ALLOW_LOW_QUALITY_MATCHES)]
        return matchers

    def compute_iou(self, boxes, anchors):
        matrix = box_iou_rotated(
            boxes[:, [0, 1, 3, 4, 6]],
            anchors[:, [0, 1, 3, 4, 6]],
        )
        return matrix

    def get_cls_targets(self, G_cls):
        """
        1. Disable ambiguous (matched to multiple classes).
        2. Replace ignore marker (-1) with binary mask.
        """
        ambiguous = G_cls.eq(1).int().sum(0) > 1
        G_cls[:, ambiguous] = -1
        M_cls = G_cls.ne(-1)
        G_cls = G_cls.clamp_(min=0)
        return G_cls, M_cls

    def _encode_diagonal(self, A_wlh):
        A_wl, A_h = A_wlh.split([2, 1], -1)
        A_norm = A_wl.norm(dim=-1, keepdim=True).expand(-1, 2)
        A_norm = torch.cat((A_norm, A_h), -1)
        return A_norm

    def get_reg_targets(self, boxes, box_idx, G_cls):
        """Standard VoxelNet-style box encoding."""
        M_reg = G_cls == 1
        A = self.anchors[M_reg]
        G = boxes[box_idx[M_reg]].cuda()
        G_xyz, G_wlh, G_yaw = G.split([3, 3, 1], -1)
        A_xyz, A_wlh, A_yaw = A.split([3, 3, 1], -1)
        A_norm = self._encode_diagonal(A_wlh)
        G_reg = torch.cat((
            (G_xyz - A_xyz) / A_norm,
            (G_wlh / A_wlh).log(),
            (G_yaw - A_yaw) % math.pi), dim=-1
        )
        M_reg = M_reg.unsqueeze(-1)
        G_reg = torch.zeros_like(self.anchors).masked_scatter_(M_reg, G_reg)
        return G_reg, M_reg

    def match_class_i(self, boxes, class_idx, full_idx, i):
        class_mask = class_idx == i
        anchors = self.anchors[i].view(-1, self.cfg.BOX_DOF)
        iou = self.compute_iou(boxes[class_mask], anchors)
        matches, labels = self.matchers[i](iou)
        if (class_mask).any():
            matches = full_idx[class_mask][matches]
        return matches, labels

    def apply_ignore_mask(self, matches, labels, box_ignore):
        """Ignore anchors matched to boxes[i] if box_ignore[i].
        E.g., boxes containing too few lidar points."""
        labels[box_ignore[matches] & (labels != -1)] = -1

    def match_all_classes(self, boxes, class_idx, box_ignore):
        """Match boxes to anchors based on IOU."""
        full_idx = torch.arange(boxes.shape[0])
        classes = range(self.cfg.NUM_CLASSES)
        matches, labels = zip(*[self.match_class_i(
            boxes, class_idx, full_idx, i) for i in classes])
        matches = torch.stack(matches).view(self.anchors.shape[:-1])
        labels = torch.stack(labels).view(self.anchors.shape[:-1])
        return matches, labels

    def forward(self, item):
        box_idx, G_cls = self.match_all_classes(
            item['boxes'].cuda(), item['class_idx'], item['box_ignore'])
        G_cls, M_cls = self.get_cls_targets(G_cls)
        G_reg, M_reg = self.get_reg_targets(item['boxes'], box_idx, G_cls)
        item.update(dict(G_cls=G_cls, G_reg=G_reg, M_cls=M_cls, M_reg=M_reg))
