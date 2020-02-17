import torch
from torch import nn


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
        anchor_sizes = [anchor['wlh'] for anchor in cfg.ANCHORS]
        anchor_radii = [anchor['radius'] for anchor in cfg.ANCHORS]
        self.anchor_sizes = torch.tensor(anchor_sizes).float()
        self.anchor_radii = torch.tensor(anchor_radii).float()
        self.anchors = self.make_anchors()

    def compute_grid_params(self):
        pixel_size = torch.tensor(self.cfg.VOXEL_SIZE[:2]) * self.cfg.STRIDES[-1]
        lower, upper = torch.tensor(self.cfg.GRID_BOUNDS).reshape(2, 3)[:, :2]
        grid_shape = ((upper - lower) / pixel_size).long()
        return lower, upper, grid_shape

    def make_anchor_sizes(self, nx, ny):
        sizes = [anchor['wlh'] for anchor in self.cfg.ANCHORS]
        sizes = torch.tensor(sizes).float()[None, None]
        sizes = sizes.expand(nx, ny, -1, -1)
        return sizes

    def make_anchor_centers(self, meshgrid_params):
        anchor_z = [anchor['center_z'] for anchor in cfg.ANCHORS]
        anchor_z = torch.tensor(anchor_z).float()
        centers = meshgrid_midpoint(*meshgrid_params)
        centers = centers.expand(-1, -1, self.cfg.NUM_CLASSES - 1, -1)
        centers[:, :, torch.arange(self.cfg.NUM_CLASSES - 1), 2] = anchor_z
        return centers

    def make_anchors(self):
        (z0, z1, nz) = 1, 1, 1
        (x0, y0), (x1, y1), (nx, ny) = self.compute_grid_params()
        centers = self.make_anchor_centers((x0, x1, nx), (y0, y1, ny), (z0, z1, nz))
        sizes = self.make_anchor_sizes(nx, ny)
        angles = centers.new_zeros((nx, ny, self.cfg.NUM_CLASSES - 1, 1))
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

    def make_cls_targets(self, inds):
        """
        Note that some negatives will be overwritten by positives.
        Last two indices are background and ignore, respectively.
        Uses one-hot encoding.
        """
        H, W, _, _ = self.anchors.shape
        targets_cls = torch.zeros((H, W, self.cfg.NUM_CLASSES + 1), dtype=torch.long)
        targets_cls[..., -1] = 1
        self.fill_negatives(targets_cls)
        self.fill_positives(targets_cls, inds)
        self.fill_ambiguous(targets_cls)
        return targets_cls

    def make_reg_targets(self, inds, boxes):
        """
        Standard VoxelNet-style box encoding.
        TODO: Angle binning.
        """
        H, W, _, _ = self.anchors.shape
        i, j, G_idx, class_idx = inds
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

    def get_targets(self, boxes, class_inds):
        # for each bounding box, see if should be assigned to cell i, j
        A_xyz, G_xyz = self.anchors[..., :3], boxes[..., :3]
        offset = G_xyz[None, None, :, :] - A_xyz[:, :, class_inds]
        match = offset.norm(dim=-1) < self.anchor_radii[class_inds]

        # positive indices
        i, j, box_idx = match.nonzero().t()
        inds = (i, j, box_idx, class_inds[box_idx])

        targets_cls = self.make_cls_targets(inds).permute(2, 1, 0)
        targets_reg = self.make_reg_targets(inds, boxes).permute(2, 3, 1, 0)
        return targets_cls, targets_reg

    def forward(self, item):
        targets_cls, targets_reg = self.get_targets(item['boxes'], item['class_ids'])
        item.update(dict(prop_targets_cls=targets_cls, prop_targets_reg=targets_reg))
