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


def torchify_anchor_attributes(cfg):
    attr = {}
    for key in ['wlh', 'center_z', 'yaw']:
        vals = [torch.tensor(anchor[key]) for anchor in cfg.ANCHORS]
        attr[key] = torch.stack(vals).float()
    return dict(attr)


class AnchorGenerator(nn.Module):
    """
    TODO: Add comment justifying unorthodox dimension ordering.
    """

    def __init__(self, cfg):
        super(AnchorGenerator, self).__init__()
        self.cfg = cfg
        self.anchor_attributes = torchify_anchor_attributes(cfg)
        self.anchors = self.make_anchors()

    def compute_grid_params(self):
        pixel_size = torch.tensor(self.cfg.VOXEL_SIZE[:2]) * self.cfg.STRIDES[-1]
        lower, upper = torch.tensor(self.cfg.GRID_BOUNDS).view(2, 3)[:, :2]
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
        centers = centers.expand(-1, -1, num_yaw, self.cfg.NUM_CLASSES, -1)
        centers[:, :, :, torch.arange(self.cfg.NUM_CLASSES), 2] = anchor_z
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
