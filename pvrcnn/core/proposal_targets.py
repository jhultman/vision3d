import torch
from torch import nn

from pvrcnn.ops import box_iou_rotated, Matcher


class ProposalTargetAssigner(nn.Module):
    """
    Match ground truth boxes to anchors by IOU.
    TODO: Simplify target_cls assignment.
    TODO: Make this run faster.
    TODO: Use F.one_hot to minimize chance of bugs.
    TODO: Include background class in targets (NUM_CLASSES + 1).
    TODO: Use Faster R-CNN style vectorized anchors to simplify indexing.
    """

    def __init__(self, cfg, anchors):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.anchors = anchors.cuda()
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

    def handle_assignment_conflicts(self, match_labels):
        """
        1. Disable ambiguous (matched to multiple classes).
        2. Clobber ignore with negative.
        3. Replace ignore -1 marker with binary mask.

        TODO: This code is hard to understand.
        """
        ambiguous = match_labels.eq(1).int().sum(0) > 1
        match_labels[:, ambiguous] = -1
        negative = match_labels.eq(0).any(0)
        positive = match_labels.eq(1).int().sum(0) == 1
        match_labels[:, negative & ~positive] = 0
        loss_mask = ~match_labels.eq(-1).all(0, keepdim=True)
        match_labels[match_labels.eq(-1)] = 0
        return loss_mask

    def _encode_diagonal(self, A_wlh):
        A_wl, A_h = A_wlh.split([2, 1], -1)
        A_norm = A_wl.norm(dim=-1, keepdim=True).expand(-1, 2)
        A_norm = torch.cat((A_norm, A_h), -1)
        return A_norm

    def get_reg_targets(self, boxes, box_idx, G_cls):
        """
        Standard VoxelNet-style box encoding.
        TODO: Angle binning, angle modulo.
        """
        A = self.anchors[G_cls == 1]
        G = boxes[box_idx[G_cls == 1]].cuda()
        G_xyz, G_wlh, G_yaw = G.split([3, 3, 1], -1)
        A_xyz, A_wlh, A_yaw = A.split([3, 3, 1], -1)
        A_norm = self._encode_diagonal(A_wlh)
        values = torch.cat((
            (G_xyz - A_xyz) / A_norm,
            (G_wlh / A_wlh).log(),
            (G_yaw - A_yaw)), dim=-1
        )
        G_reg = torch.zeros_like(self.anchors)
        G_reg[G_cls == 1] = values
        M_reg = G_cls[:, :-1, ..., None].sum(1, keepdim=True).float()
        return G_reg, M_reg

    def get_matches(self, boxes, class_idx):
        """Match boxes to anchors based on IOU."""
        full_idx = torch.arange(boxes.shape[0])
        matches, match_labels = [], []
        for i in range(self.cfg.NUM_CLASSES):
            if not (class_idx == i).any():
                continue
            anchors_i = self.anchors[i].view(-1, self.cfg.BOX_DOF)
            iou = self.compute_iou(boxes[class_idx == i].cuda(), anchors_i)
            _matches, _match_labels = self.matchers[i](iou)
            matches += [full_idx[class_idx == i][_matches]]
            match_labels += [_match_labels]
        matches = torch.stack(matches).view(self.anchors.shape[:-1])
        match_labels = torch.stack(match_labels).view(self.anchors.shape[:-1])
        return matches, match_labels

    def forward(self, item):
        boxes, class_idx = item['boxes'], item['class_idx']
        box_idx, G_cls = self.get_matches(boxes, class_idx)
        M_cls = self.handle_assignment_conflicts(G_cls).float()
        G_reg, M_reg = self.get_reg_targets(boxes, box_idx, G_cls)
        item.update(dict(G_cls=G_cls.float(), G_reg=G_reg, M_cls=M_cls, M_reg=M_reg))
