import numpy as np


def points_in_convex_polygon(points, polygon, ccw=True):
    """points (N, 2) | polygon (M, V, 2) | mask (N, M)"""
    polygon_roll = np.roll(polygon, shift=1, axis=1)
    polygon_side = (-1) ** ccw * (polygon - polygon_roll)[None]
    vertex_to_point = polygon[None] - points[:, None, None]
    mask = (np.cross(polygon_side, vertex_to_point) > 0).all(2)
    return mask


def box3d_to_bev_corners(boxes):
    """
    :boxes np.ndarray shape (N, 7)
    :corners np.ndarray shape (N, 4, 2) (ccw)
    """
    xy, _, wl, _, yaw = np.split(boxes, [2, 3, 5, 6], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
    corners = 0.5 * np.r_[-1, -1, +1, -1, +1, +1, -1, +1]
    corners = (wl[:, None] * corners.reshape(4, 2))
    corners = np.einsum('ijk,imk->imj', R, corners) + xy[:, None]
    return corners


class PointsInCuboids:
    """Takes ~10ms for each scene."""

    def __init__(self, points):
        self.points = points

    def _height_threshold(self, boxes):
        """Filter to z slice."""
        z1 = self.points[:, None, 2]
        z2, h = boxes[:, [2, 5]].T
        mask = (z1 > z2 - h / 2) & (z1 < z2 + h / 2)
        return mask

    def _get_mask(self, boxes):
        polygons = box3d_to_bev_corners(boxes)
        mask = self._height_threshold(boxes)
        mask &= points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return list of points in each box."""
        mask = self._get_mask(boxes).T
        points = list(map(self.points.__getitem__, mask))
        return points


class PointsNotInRectangles(PointsInCuboids):

    def _get_mask(self, boxes):
        polygons = box3d_to_bev_corners(boxes)
        mask = points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return array of points not in any box."""
        mask = ~self._get_mask(boxes).any(1)
        return self.points[mask]
