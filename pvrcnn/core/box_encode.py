import torch
import math


def _anchor_diagonal(A_wlh):
    """Reference: VoxelNet."""
    A_wl, A_h = A_wlh.split([2, 1], -1)
    A_norm = A_wl.norm(dim=-1, keepdim=True).expand_as(A_wl)
    A_norm = torch.cat((A_norm, A_h), dim=-1)
    return A_norm


def decode(deltas, anchors):
    """Both inputs of shape (*, 7)."""
    P_xyz, P_wlh, P_yaw = deltas.split([3, 3, 1], -1)
    A_xyz, A_wlh, A_yaw = anchors.split([3, 3, 1], -1)
    A_norm = _anchor_diagonal(A_wlh)
    boxes = torch.cat((
        (P_xyz * A_norm + A_xyz),
        (P_wlh.exp() * A_wlh),
        (P_yaw + A_yaw)), dim=-1
    )
    return boxes


def encode(boxes, anchors):
    """Both inputs of shape (*, 7)."""
    G_xyz, G_wlh, G_yaw = boxes.split([3, 3, 1], -1)
    A_xyz, A_wlh, A_yaw = anchors.split([3, 3, 1], -1)
    A_norm = _anchor_diagonal(A_wlh)
    deltas = torch.cat((
        (G_xyz - A_xyz) / A_norm,
        (G_wlh / A_wlh).log(),
        (G_yaw - A_yaw) % math.pi), dim=-1
    )
    return deltas
