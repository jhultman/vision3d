import pickle
import numpy as np
import torch
import os
from tqdm import tqdm
from copy import deepcopy
import os.path as osp
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner
from .kitti_utils import read_calib, read_label, read_velo, filter_camera_fov
from .augmentation import ChainedAugmentation, DatabaseBuilder


class AnnotationLoader:
    """Load annotations if exist, else create."""

    def __init__(self, cfg, inds, split='val'):
        super(AnnotationLoader, self).__init__()
        self.CACHEDIR = cfg.DATA.CACHEDIR
        self.ROOTDIR = cfg.DATA.ROOTDIR
        self.inds, self.split = inds, split
        self.load_annotations()
        if split == 'train':
            DatabaseBuilder(cfg, self.annotations)

    def read_cached_annotations(self):
        fpath = osp.join(self.CACHEDIR, f'{self.split}.pkl')
        print(f'Loading cached annotations: {fpath}')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)

    def cache_annotations(self):
        fpath = osp.join(self.CACHEDIR, f'{self.split}.pkl')
        print(f'Caching annotations: {fpath}')
        with open(fpath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def load_annotations(self):
        try:
            self.read_cached_annotations()
        except FileNotFoundError:
            os.makedirs(self.CACHEDIR, exist_ok=True)
            self.create_annotations()
            self.crop_points()
            self.cache_annotations()

    def crop_points(self):
        """Limit points to camera FOV (KITTI-specific)."""
        dir_new = osp.join(self.ROOTDIR, 'velodyne_reduced')
        if osp.isdir(dir_new):
            return print(f'Found existing reduced points: {dir_new}')
        os.makedirs(dir_new, exist_ok=False)
        for anno in tqdm(self.annotations.values(), desc='Filtering points'):
            basename = osp.basename(anno['velo_path'])
            path_old = osp.join(self.ROOTDIR, 'velodyne', basename)
            points = filter_camera_fov(anno['calib'], read_velo(path_old))
            points.astype(np.float32).tofile(osp.join(dir_new, basename))

    def _path_helper(self, fdir, idx, end):
        fpath = osp.join(self.ROOTDIR, fdir, f'{idx:06d}.{end}')
        return fpath

    def create_annotations(self):
        self.annotations = dict()
        for idx in tqdm(self.inds, desc='Creating annotations'):
            item = dict(
                velo_path=self._path_helper('velodyne_reduced', idx, 'bin'),
                objects=read_label(self._path_helper('label_2', idx, 'txt')),
                calib=read_calib(self._path_helper('calib', idx, 'txt')), idx=idx,
            )
            self.numpify_objects(item)
            self.annotations[idx] = item

    def _numpify_object(self, obj, calib):
        """Converts from camera to velodyne frame."""
        xyz = calib.C2V @ np.r_[calib.R0 @ obj.t, 1]
        box = np.r_[xyz, obj.w, obj.l, obj.h, -obj.ry]
        obj = dict(box=box, class_idx=obj.class_idx)
        return obj

    def numpify_objects(self, item):
        objects = [self._numpify_object(
            obj, item['calib']) for obj in item['objects']]
        item['boxes'] = np.stack([obj['box'] for obj in objects])
        item['class_idx'] = np.r_[[obj['class_idx'] for obj in objects]]
        item.pop('objects')


class KittiDataset(Dataset):

    def __init__(self, cfg, split='val'):
        super(KittiDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.load_annotations(cfg)

    def __len__(self):
        return len(self.inds)

    def load_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.SPLITDIR, f'{self.split}.txt')
        self.inds = np.loadtxt(fpath, dtype=np.int32).tolist()
        loader = AnnotationLoader(cfg, self.inds, self.split)
        self.annotations = loader.annotations

    def filter_bad_objects(self, item):
        class_idx = item['class_idx'][:, None]
        _, wlh, _ = np.split(item['boxes'], [3, 6], 1)
        keep = ((class_idx != -1) & (wlh > 0)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def filter_out_of_bounds(self, item):
        xyz, _, _ = np.split(item['boxes'], [3, 6], 1)
        lower, upper = np.split(self.cfg.GRID_BOUNDS, [3])
        keep = ((xyz >= lower) & (xyz <= upper)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def to_torch(self, item):
        item['points'] = np.float32(item['points'])
        item['boxes'] = torch.FloatTensor(item['boxes'])
        item['class_idx'] = torch.LongTensor(item['class_idx'])
        item['box_ignore'] = torch.full_like(
            item['class_idx'], False, dtype=torch.bool)

    def drop_keys(self, item):
        for key in ['velo_path', 'calib']:
            item.pop(key)

    def preprocessing(self, item):
        self.to_torch(item)

    def __getitem__(self, idx):
        item = deepcopy(self.annotations[self.inds[idx]])
        item['points'] = read_velo(item['velo_path'])
        self.preprocessing(item)
        self.drop_keys(item)
        return item


class KittiDatasetTrain(KittiDataset):
    """TODO: This class should certainly not need access to
        anchors. Find better place to instantiate target assigner."""

    def __init__(self, cfg):
        super(KittiDatasetTrain, self).__init__(cfg, split='train')
        self.augmentation = ChainedAugmentation(cfg)
        self.target_assigner = ProposalTargetAssigner(cfg)

    def preprocessing(self, item):
        """Applies augmentation and assigns targets."""
        np.random.shuffle(item['points'])
        self.filter_bad_objects(item)
        points, boxes, class_idx = self.augmentation(
            item['points'], item['boxes'], item['class_idx'])
        item.update(dict(points=points, boxes=boxes, class_idx=class_idx))
        self.filter_out_of_bounds(item)
        self.to_torch(item)
        self.target_assigner(item)
