import pickle
import itertools
import numpy as np
import os.path as osp
from tqdm import tqdm
from collections import defaultdict

from .kitti_utils import read_velo


def points_in_convex_polygon(points, polygon, ccw=True):
    """points (N, 2) | polygon (M, V, 2) | mask (N, M)"""
    polygon_roll = np.roll(polygon, shift=1, axis=1)
    polygon_side = (-1) ** ccw * (polygon - polygon_roll)[None]
    vertex_to_point = polygon[None] - points[:, None, None]
    mask = (np.cross(polygon_side, vertex_to_point) > 0).all(2)
    return mask


def center_to_corner_box2d(boxes):
    """
    Corners returned counter-clockwise.
    TODO: Document input dimensions.
    """
    xy, _, wl, _, yaw = np.split(boxes, [2, 3, 5, 6], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
    corners = 0.5 * np.r_[-1, -1, +1, -1, +1, +1, -1, +1]
    corners = (wl[:, None] * corners.reshape(4, 2))
    corners = np.einsum('ijk,imk->imj', R, corners) + xy[:, None]
    return corners


class DatabaseBuilder:

    def __init__(self, cfg, annotations):
        self.cfg = cfg
        self.fpath = osp.join(cfg.DATA.CACHEDIR, 'database.pkl')
        self._build(annotations)

    def _build(self, annotations):
        if osp.isfile(self.fpath):
            print(f'Found cached database: {self.fpath}')
            return
        database = defaultdict(list)
        for item in tqdm(annotations.values(), desc='Building database'):
            for key, val in zip(*self._process_item(item)):
                database[key] += [val]
        self._save_database(dict(database))

    def _demean(self, points, boxes):
        """Subtract box center (birds eye view)."""
        _points, _boxes = [], []
        for points_i, box_i in zip(points, boxes):
            center, zwlhr = np.split(box_i, [2])
            xy, zi = np.split(points_i, [2], 1)
            _points += [np.concatenate((xy - center, zi), 1)]
            _boxes += [np.concatenate((0 * center, zwlhr))]
        return _points, _boxes

    def _process_item(self, item):
        """Retrieve points in each box in scene."""
        points = read_velo(item['velo_path'])
        class_idx, boxes = item['class_idx'], item['boxes']
        points = PointsInCuboids(points)(boxes)
        keep = [len(p) > self.cfg.AUG.MIN_NUM_SAMPLE_PTS for p in points]
        class_idx, points, boxes = [
            itertools.compress(t, keep) for t in (class_idx, points, boxes)]
        points, boxes = self._demean(points, boxes)
        samples = [dict(points=p, box=b) for (p, b) in zip(points, boxes)]
        return class_idx, samples

    def _save_database(self, database):
        with open(self.fpath, 'wb') as f:
            pickle.dump(database, f)


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
        polygons = center_to_corner_box2d(boxes)
        mask = self._height_threshold(boxes)
        mask &= points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return list of points in each box."""
        mask = self._get_mask(boxes).T
        points = list(map(self.points.__getitem__, mask))
        return points


class PointsNotInCuboids(PointsInCuboids):
    """
    TODO: This shares no methods with PointsInCuboids.
    """

    def _get_mask(self, boxes):
        polygons = center_to_corner_box2d(boxes)
        mask = points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return array of points not in any box."""
        mask = ~self._get_mask(boxes).any(1)
        return self.points[mask]
