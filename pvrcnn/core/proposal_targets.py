import torch
from torch import nn
from collections import defaultdict

from pvrcnn.ops import rotated_iou


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
    for key in ['wlh', 'center_z', 'yaw', 'iou_thresh']:
        vals = [torch.tensor(anchor[key]) for anchor in cfg.ANCHORS]
        attr[key] = torch.stack(vals).float()
    return dict(attr)


class ProposalTargetAssigner(nn.Module):
    """
    Simple target assignment algorithm based on Sparse-to-Dense.
    Anchors considered positive if within category-specific
    max spherical radius of box center.

    TODO: Move anchor generation elsewhere, pass in constructor.
    """

    def __init__(self, cfg):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.anchor_attributes = torchify_anchor_attributes(cfg)
        self.anchors = self.make_anchors()

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
        return anchors

    def fill_negatives(self, targets_cls):
        H, W, _ = targets_cls.shape
        M = self.cfg.TRAIN.PROPOSAL_NUM_NEGATIVES
        i = torch.randint(H, (M,), dtype=torch.long)
        j = torch.randint(W, (M,), dtype=torch.long)
        targets_cls[i, j, -1] = 0
        targets_cls[i, j, -2] = 1

    def fill_positives(self, targets_cls, inds):
        i, j, _, h = inds
        targets_cls[i, j, h] = 1
        targets_cls[i, j, -2:] = 0

    def fill_ambiguous(self, targets_cls):
        """Disables positives matched to multiple classes."""
        ambiguous = targets_cls.int().sum(2) > 1
        targets_cls[ambiguous][:, :-1] = 0
        targets_cls[ambiguous][:, -1] = 1

    def make_targets_cls(self, pos_inds, neg_inds):
        """
        Note that some negatives will be overwritten by positives.
        Last two indices are background and ignore, respectively.
        Uses one-hot encoding.
        """
        H, W, _, _ = self.anchors.shape
        targets_cls = torch.zeros((H, W, self.cfg.NUM_CLASSES + 1), dtype=torch.long)
        targets_cls[..., -1] = 1
        self.fill_negatives(targets_cls, neg_inds)
        self.fill_positives(targets_cls, pos_inds)
        self.fill_ambiguous(targets_cls)
        return targets_cls

    def make_targets_reg(self, pos_inds, boxes):
        """
        Standard VoxelNet-style box encoding.
        TODO: Angle binning.
        """
        H, W, _, _ = self.anchors.shape
        i, j, G_idx, class_idx = pos_inds
        A_idx = (i, j, class_idx)
        targets_reg = torch.zeros((H, W, self.cfg.BOX_DOF), dtype=torch.float32)
        G_xyz, G_wlh, G_yaw = boxes[G_idx].split([3, 3, 1], -1)
        A_xyz, A_wlh, A_yaw = self.anchors[A_idx].split([3, 3, 1], -1)
        values = torch.cat((
            (G_xyz - A_xyz),
            (G_wlh - A_wlh) / A_wlh,
            (G_yaw - A_yaw)), dim=-1
        )
        targets_reg[i, j] = values
        return targets_reg

    def compute_iou_matrix(self, anchors, boxes):
        matrix = rotated_iou(
            anchors[:, [0, 1, 3, 4, 6]],
            boxes[:, [0, 1, 3, 4, 6]],
        )
        return matrix

    def make_target_indices(self, match_matrix, class_idx):
        anchor_idx, box_idx = match_matrix.nonzero().t()
        indices = (i, j, box_idx, class_idx[box_idx])
        return indices

    def single_class_assign(self, anchors, boxes, thresh):
        iou = self.compute_iou_matrix(anchors, boxes)
        negative, positive = iou < thresh[0], iou > thresh[1]
        pos_inds = self.make_target_indices(positive)
        neg_inds = self.make_target_indices(negative)
        targets_cls = self.make_targets_cls(anchors, pos_inds, neg_inds)
        targets_reg = self.make_targets_reg(anchors, pos_inds, boxes)
        return targets_cls, targets_reg

    def to_multiclass(self, targets):
        pass

    def forward(self, item):
        boxes, class_idx = item['boxes'], item['class_idx']
        item['prop_targets_cls'], item['prop_targets_reg']
        targets = []
        for i in range(n_cls):
            cls__boxes = boxes[class_idx == i]
            thresh = self.anchor_attributes['iou_thresh'][i]
            cls_anchors = self.anchors[:, :, :, i].view(-1, self.cfg.BOX_DOF)
            targets += [self.single_class_assign(cls_anchors, cls_boxes, thresh)]
        targets = self.to_multiclass(targets)
        return matrix
