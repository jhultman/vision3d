import torch
import math
from torch import nn

from pvrcnn.ops import box_iou_rotated, Matcher
from .anchor_generator import AnchorGenerator
from .box_encode import encode


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
        Clamps ignore to 0 and represents with binary mask.
        Note: allows anchor to be matched to multiple classes.
        """
        M_cls = G_cls.ne(-1)
        G_cls = G_cls.clamp_(min=0)
        return G_cls, M_cls

    def get_reg_targets(self, boxes, box_idx, G_cls):
        """Standard VoxelNet-style box encoding."""
        M_reg = G_cls == 1
        G_reg = encode(boxes[box_idx[M_reg]], self.anchors[M_reg])
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
        full_idx = torch.arange(boxes.shape[0], device=boxes.device)
        classes = range(self.cfg.NUM_CLASSES)
        matches, labels = zip(*[self.match_class_i(
            boxes, class_idx, full_idx, i) for i in classes])
        matches = torch.stack(matches).view(self.anchors.shape[:-1])
        labels = torch.stack(labels).view(self.anchors.shape[:-1])
        return matches, labels

    def to_device(self, item):
        """Move items to anchors.device for fast rotated IOU."""
        keys = ['boxes', 'class_idx', 'box_ignore']
        items = [item[k].to(self.anchors.device) for k in keys]
        return items

    def forward(self, item):
        boxes, class_idx, box_ignore = self.to_device(item)
        box_idx, G_cls = self.match_all_classes(boxes, class_idx, box_ignore)
        G_cls, M_cls = self.get_cls_targets(G_cls)
        G_reg, M_reg = self.get_reg_targets(boxes, box_idx, G_cls)
        item.update(dict(G_cls=G_cls, G_reg=G_reg, M_cls=M_cls, M_reg=M_reg))
