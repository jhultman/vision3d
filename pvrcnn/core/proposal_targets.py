import torch
from torch import nn

from pvrcnn.ops import rotated_iou
from pvrcnn.thirdparty import Matcher, subsample_labels


class ProposalTargetAssigner(nn.Module):
    """
    Match ground truth boxes to anchors by IOU.
    TODO: Refactor target_cls assignment -- too much scatter/indexing.
    """

    def __init__(self, cfg, anchors):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.anchors = anchors
        self.matchers = self.build_matchers(cfg)

    def build_matchers(self, cfg):
        matchers = []
        for anchor in cfg.ANCHORS:
            matchers += [Matcher(anchor['iou_thresh'],
                [0, -1, +1], cfg.ALLOW_LOW_QUALITY_MATCHES)]
        return matchers

    def compute_iou_matrix(self, boxes, anchors):
        matrix = rotated_iou(
            boxes[:, [0, 1, 3, 4, 6]],
            anchors[:, [0, 1, 3, 4, 6]],
        )
        return matrix

    def resample_pos_neg(self, match_labels):
        """Tries to sample positives and negatives 50/50."""
        match_labels = match_labels.view(-1)
        pos_idx, neg_idx = subsample_labels(
            match_labels, self.cfg.NUM_PROPOSAL_SAMPLE, 0.5, 0
        )
        match_labels.fill_(-1)
        match_labels.scatter_(0, pos_idx, 1)
        match_labels.scatter_(0, neg_idx, 0)

    def handle_assignment_conflicts(self, match_labels):
        """
        1. Disable ambiguous (matched to multiple classes).
        2. Clobber ignore with negative.
        3. Replace ignore -1 marker with binary mask.
        """
        ambiguous = match_labels.eq(1).int().sum(0) > 1
        match_labels[:, ambiguous] = -1
        negative = match_labels.eq(0).any(0) > 0
        positive = match_labels.eq(1).int().sum(0) == 1
        match_labels[:, negative & ~positive] = 0
        ignore_mask = match_labels.eq(-1).all(0)
        match_labels[match_labels.eq(-1)] = 0
        return ignore_mask

    def get_cls_targets(self, match_labels):
        self.resample_pos_neg(match_labels)
        ignore_mask = self.handle_assignment_conflicts(match_labels)
        onehot = torch.cat((match_labels, ignore_mask[None].type_as(match_labels))).float()
        return onehot

    def get_reg_targets(self, boxes, matches, match_labels):
        """
        Standard VoxelNet-style box encoding.
        TODO: Angle binning.
        """
        A = self.anchors[match_labels == 1]
        G = boxes[matches[match_labels == 1]]
        G_xyz, G_wlh, G_yaw = G.split([3, 3, 1], -1)
        A_xyz, A_wlh, A_yaw = A.split([3, 3, 1], -1)
        values = torch.cat((
            (G_xyz - A_xyz),
            (G_wlh - A_wlh) / A_wlh,
            (G_yaw - A_yaw)), dim=-1
        )
        targets = torch.zeros_like(self.anchors)
        targets[match_labels == 1] = values
        return targets

    def get_matches(self, boxes, class_idx):
        """Match boxes to anchors based on IOU."""
        n_cls, n_yaw, ny, nx, _ = self.anchors.shape
        all_matches = torch.full((n_cls, n_yaw, ny, nx), -1, dtype=torch.long)
        all_match_labels = torch.full((n_cls, n_yaw, ny, nx), -1, dtype=torch.int8)
        for i in range(self.cfg.NUM_CLASSES - 1):
            class_mask = class_idx == i
            if not (class_mask).any():
                continue
            anchors_i = self.anchors[i].view(-1, self.cfg.BOX_DOF)
            iou_matrix = self.compute_iou_matrix(boxes[class_mask].cuda(), anchors_i.cuda())
            matches, match_labels = self.matchers[i](iou_matrix)
            all_matches[i].view(-1)[:] = matches
            all_match_labels[i].view(-1)[:] = match_labels
        return all_matches, all_match_labels

    def forward(self, item):
        boxes, class_idx = item['boxes'], item['class_idx']
        matches, match_labels = self.get_matches(boxes, class_idx)
        item['proposal_targets_cls'] = self.get_cls_targets(match_labels)
        item['proposal_targets_reg'] = self.get_reg_targets(boxes, matches, match_labels)
