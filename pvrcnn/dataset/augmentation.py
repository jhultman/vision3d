import pickle
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import compress

from .kitti_utils import read_velo
from pvrcnn.ops import box_iou_rotated
from pvrcnn.core.geometry import (
    points_in_convex_polygon,
    PointsNotInRectangles,
    PointsInCuboids,
)


class Augmentation:

    def __init__(self, cfg):
        self.cfg = cfg

    def uniform(self, *args):
        return np.float32(np.random.uniform(*args))

    def __call__(self, points, boxes, *args):
        raise NotImplementError


class ChainedAugmentation(Augmentation):

    def __init__(self, cfg):
        super(ChainedAugmentation, self).__init__(cfg)
        self.augmentations = [
            SampleAugmentation(cfg),
            FlipAugmentation(cfg),
            ScaleAugmentation(cfg),
            RotateAugmentation(cfg),
        ]

    def __call__(self, points, boxes, class_idx):
        if self.cfg.AUG.DATABASE_SAMPLE:
            points, boxes, class_idx = self.augmentations[0](
                points, boxes, class_idx)
        for aug in self.augmentations[1:]:
            points, boxes = aug(points, boxes)
        return points, boxes, class_idx


class RotateAugmentation(Augmentation):

    def rotate(self, theta, xy):
        """Right-multiplies by transpose for convenience."""
        c, s = np.cos(theta), np.sin(theta)
        points = xy @ np.r_[c, s, -s, c].reshape(2, 2)
        return points

    def _split_rotate_points(self, theta, points):
        xy, zi = np.split(points, [2], 1)
        xy = self.rotate(theta, xy)
        points = np.concatenate((xy, zi), 1)
        return points

    def _split_rotate_boxes(self, theta, boxes):
        xy, zwlh, rz = np.split(boxes, [2, 6], 1)
        xy = self.rotate(theta, xy)
        boxes = np.concatenate((xy, zwlh, rz + theta), 1)
        return boxes

    def __call__(self, points, boxes):
        theta = self.uniform(*self.cfg.AUG.GLOBAL_ROTATION)
        points = self._split_rotate_points(theta, points)
        boxes = self._split_rotate_boxes(theta, boxes)
        return points, boxes


class FlipAugmentation(Augmentation):

    def _split_flip_points(self, points):
        x, y, zi = np.split(points, [1, 2], 1)
        points = np.concatenate((x, -y, zi), 1)
        return points

    def _split_flip_boxes(self, boxes):
        x, y, zwlh, rz = np.split(boxes, [1, 2, 6], 1)
        boxes = np.concatenate((x, -y, zwlh, -rz), 1)
        return boxes

    def __call__(self, points, boxes):
        if np.random.rand() < 0.5 or not self.cfg.AUG.FLIP_HORIZONTAL:
            return points, boxes
        points = self._split_flip_points(points)
        boxes = self._split_flip_boxes(boxes)
        return points, boxes


class ScaleAugmentation(Augmentation):

    def _split_scale_points(self, factor, points):
        xyz, i = np.split(points, [3], 1)
        points = np.concatenate((factor * xyz, i), 1)
        return points

    def _split_scale_boxes(self, factor, boxes):
        xyzwlh, rz = np.split(boxes, [6], 1)
        boxes = np.concatenate((factor * xyzwlh, rz), 1)
        return boxes

    def __call__(self, points, boxes):
        factor = self.uniform(*self.cfg.AUG.GLOBAL_SCALE)
        points = self._split_scale_points(factor, points)
        boxes = self._split_scale_boxes(factor, boxes)
        return points, boxes


class SampleAugmentation(Augmentation):
    """Pastes samples from database into scene."""

    def __init__(self, cfg):
        super(SampleAugmentation, self).__init__(cfg)
        self._load_database(cfg)

    def _load_database(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, 'database.pkl')
        with open(fpath, 'rb') as f:
            self.database = pickle.load(f)

    def draw_samples(self):
        """Draw samples from each class."""
        samples = []
        for class_idx in range(self.cfg.NUM_CLASSES):
            indices = np.random.choice(
                len(self.database[class_idx]),
                self.cfg.AUG.NUM_SAMPLE_OBJECTS[class_idx],
            ).tolist()
            _samples = [self.database[class_idx][i] for i in indices]
            list(map(lambda s: s.update(dict(class_idx=class_idx)), _samples))
            samples += list(_samples)
        return samples

    def filter_collisions(self, boxes, sample_boxes):
        """Remove samples with BEV iou > 0."""
        N = boxes.shape[0]
        boxes = torch.cat((
            torch.from_numpy(boxes),
            torch.from_numpy(sample_boxes),
        )).cuda().float()[:, [0, 1, 3, 4, 6]]
        iou = box_iou_rotated(boxes, boxes).cpu().numpy()
        mask = (iou > 1e-2).sum(1)[N:] == 1
        return mask

    def _translate_points(self, points, position):
        """Apply box translations to corresponding points."""
        _points = []
        for points_i, position_i in zip(points, position):
            xy, zi = np.split(points_i, [2], 1)
            _points += [np.concatenate((xy + position_i, zi), 1)]
        return _points

    def random_translate(self, samples):
        """Apply random xy-translation to sampled boxes."""
        boxes = samples['boxes']
        lower, upper = np.r_[self.cfg.GRID_BOUNDS].reshape(2, 3)[:, :2]
        position = np.random.rand(len(boxes), 2) * (upper - lower) + lower
        samples['boxes'] = boxes + np.pad(position, ((0, 0), (0, 5)))
        samples['points'] = self._translate_points(samples['points'], position)
        return boxes, position

    def reorganize_samples(self, samples):
        """Convert list of dicts to dict of lists."""
        _samples = dict()
        for key in ['points', 'box', 'class_idx']:
            _samples[key] = [s[key] for s in samples]
        _samples['boxes'] = np.stack(_samples.pop('box'))
        return _samples

    def mask_samples(self, samples, mask):
        """Remove samples participating in collisions."""
        samples['boxes'] = samples['boxes'][mask]
        samples['points'] = list(compress(samples['points'], mask))
        samples['class_idx'] = list(compress(samples['class_idx'], mask))

    def cat_samples(self, samples, points, boxes, class_idx):
        points = np.concatenate([points] + samples['points'])
        boxes = np.concatenate((boxes, samples['boxes']))
        class_idx = np.concatenate((class_idx, samples['class_idx']))
        return points, boxes, class_idx

    def __call__(self, points, boxes, class_idx):
        samples = self.draw_samples()
        samples = self.reorganize_samples(samples)
        self.random_translate(samples)
        mask = self.filter_collisions(boxes, samples['boxes'])
        self.mask_samples(samples, mask)
        points = PointsNotInRectangles(points)(samples['boxes'])
        points, boxes, class_idx = self.cat_samples(
            samples, points, boxes, class_idx)
        return points, boxes, class_idx


class DatabaseBuilder:
    """Builds cached database for SampleAugmentation."""

    def __init__(self, cfg, annotations):
        self.cfg = cfg
        self.fpath = osp.join(cfg.DATA.CACHEDIR, 'database.pkl')
        if osp.isfile(self.fpath):
            print(f'Found cached database: {self.fpath}')
            return
        self._build(annotations)

    def _build(self, annotations):
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
            compress(t, keep) for t in (class_idx, points, boxes)]
        points, boxes = self._demean(points, boxes)
        samples = [dict(points=p, box=b) for (p, b) in zip(points, boxes)]
        return class_idx, samples

    def _save_database(self, database):
        with open(self.fpath, 'wb') as f:
            pickle.dump(database, f)
