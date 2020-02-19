import torch
from torch import nn
from collections import defaultdict

from pvrcnn.ops import rotated_iou
from pvrcnn.thirdparty import Matcher, subsample_labels


def _linspace_midpoint(x0, x1, nx):
    """
    Mimics np.linspace with endpoint=False except
    samples fall in middle of bin instead of end.
    """
    dx = (x1 - x0) / nx
    x = torch.linspace(x0, x1 - dx, nx) + dx / 2
    return x


def meshgrid_midpoint(*arrays):
    """Customized meshgrid to use the above."""
    spaces = [_linspace_midpoint(*x) for x in arrays]
    grid = torch.stack(torch.meshgrid(spaces), -1)
    return grid


def torchify_anchor_attributes(cfg):
    attr = {}
    for key in ['wlh', 'center_z', 'yaw']:
        vals = [torch.tensor(anchor[key]) for anchor in cfg.ANCHORS]
        attr[key] = torch.stack(vals).float()
    return dict(attr)


class ProposalTargetAssigner(nn.Module):
    """
    Simple target assignment algorithm based on Sparse-to-Dense.
    Anchors considered positive if within category-specific
    max spherical radius of box center.

    TODO: Move anchor generation elsewhere, pass in constructor.
    TODO: Refactor target_cls assignment -- too much memory copy.
    """

    def __init__(self, cfg):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.matchers = self.build_matchers(cfg)
        self.anchor_attributes = torchify_anchor_attributes(cfg)
        self.anchors = self.make_anchors()

    def build_matchers(self, cfg):
        matchers = []
        for anchor in cfg.ANCHORS:
            matchers += [Matcher(anchor['iou_thresh'],
                [0, -1, +1], cfg.ALLOW_LOW_QUALITY_MATCHES)]
        return matchers

    def compute_grid_params(self):
        pixel_size = torch.tensor(self.cfg.VOXEL_SIZE[:2]) * self.cfg.STRIDES[-1]
        lower, upper = torch.tensor(self.cfg.GRID_BOUNDS).reshape(2, 3)[:, :2]
        grid_shape = ((upper - lower) / pixel_size).long()
        return lower, upper, grid_shape

    def make_anchor_sizes(self, nx, ny):
        num_yaw = self.anchor_attributes['yaw'].shape[1]
        sizes = self.anchor_attributes['wlh'][None, None, None]
        sizes = sizes.expand(nx, ny, num_yaw, -1, -1)
        return sizes

    def make_anchor_centers(self, meshgrid_params):
        num_yaw = self.anchor_attributes['yaw'].shape[1]
        anchor_z = self.anchor_attributes['center_z']
        centers = meshgrid_midpoint(*meshgrid_params)[:, :, None]
        centers = centers.expand(-1, -1, num_yaw, self.cfg.NUM_CLASSES - 1, -1)
        centers[:, :, :, torch.arange(self.cfg.NUM_CLASSES - 1), 2] = anchor_z
        return centers

    def make_anchor_angles(self, nx, ny):
        yaw = self.anchor_attributes['yaw'].T[None, None, ..., None]
        yaw = yaw.expand(nx, ny, -1, -1, -1)
        return yaw

    def make_anchors(self):
        (z0, z1, nz) = 1, 1, 1
        (x0, y0), (x1, y1), (nx, ny) = self.compute_grid_params()
        centers = self.make_anchor_centers([(x0, x1, nx), (y0, y1, ny), (z0, z1, nz)])
        sizes = self.make_anchor_sizes(nx, ny)
        angles = self.make_anchor_angles(nx, ny)
        anchors = torch.cat((centers, sizes, angles), dim=-1)
        anchors = anchors.permute(3, 2, 1, 0, 4).contiguous()
        return anchors

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

    def fill_onehot(self, onehot, match_labels, ignore_mask):
        negative = match_labels.eq(0).all(0) & ~ignore_mask
        onehot[:-2] = match_labels
        onehot[-2] = negative
        onehot[-1] = ignore_mask

    def get_cls_targets(self, match_labels):
        self.resample_pos_neg(match_labels)
        ignore_mask = self.handle_assignment_conflicts(match_labels)
        n_cls, n_yaw, ny, nx = match_labels.shape
        onehot = match_labels.new_zeros((n_cls + 2, n_yaw, ny, nx))
        self.fill_onehot(onehot, match_labels, ignore_mask)
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
