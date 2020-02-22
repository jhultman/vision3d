from tqdm import tqdm
import pickle
import numpy as np
import torch
from copy import deepcopy
import os.path as osp
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner, AnchorGenerator
from .kitti_utils import read_calib, read_label, read_velo
from .augmentation import ChainedAugmentation
from .database_sampler import DatabaseBuilder


class KittiDataset(Dataset):
    """
    TODO: This class should certainly not need
    access to anchors. Find better place to
    instantiate target assigner.
    """

    def __init__(self, cfg, split):
        super(KittiDataset, self).__init__()
        self.split = split
        self.rootdir = cfg.DATA.ROOTDIR
        self.load_annotations(cfg)
        if split == 'train':
            DatabaseBuilder(cfg, self.annotations)
            anchors = AnchorGenerator(cfg).anchors
            self.target_assigner = ProposalTargetAssigner(cfg, anchors)
            self.augmentation = ChainedAugmentation(cfg)
        self.cfg = cfg

    def __len__(self):
        return len(self.inds)

    def read_splitfile(self, cfg):
        fpath = osp.join(cfg.DATA.SPLITDIR, f'{self.split}.txt')
        self.inds = np.loadtxt(fpath, dtype=np.int32).tolist()

    def try_read_cached_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        if not osp.isfile(fpath):
            return False
        print(f'Found cached annotations: {fpath}')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        return True

    def cache_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def create_annotation(self, idx, cfg):
        velo_path = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced', f'{idx:06d}.bin')
        calib = read_calib(osp.join(cfg.DATA.ROOTDIR, 'calib', f'{idx:06d}.txt'))
        objects = read_label(osp.join(cfg.DATA.ROOTDIR, 'label_2', f'{idx:06d}.txt'))
        item = dict(velo_path=velo_path, calib=calib, objects=objects, idx=idx)
        self.make_simple_objects(item)
        return item

    def load_annotations(self, cfg):
        self.read_splitfile(cfg)
        if self.try_read_cached_annotations(cfg):
            return
        print('Generating annotations...')
        self.annotations = dict()
        for idx in tqdm(self.inds):
            self.annotations[idx] = self.create_annotation(idx, cfg)
        self.cache_annotations(cfg)

    def make_simple_object(self, obj, calib):
        """Convert coordinates to velodyne frame."""
        xyz = calib.R0 @ obj.t
        xyz = calib.C2V @ np.r_[xyz, 1]
        wlh = np.r_[obj.w, obj.l, obj.h]
        rz = np.r_[-obj.ry]
        box = np.r_[xyz, wlh, rz]
        obj = dict(box=box, class_idx=obj.class_idx)
        return obj

    def make_simple_objects(self, item):
        objects = [self.make_simple_object(
            obj, item['calib']) for obj in item['objects']]
        item['boxes'] = np.stack([obj['box'] for obj in objects])
        item['class_idx'] = np.r_[[obj['class_idx'] for obj in objects]]

    def drop_keys(self, item):
        for key in ['velo_path', 'objects', 'calib']:
            item.pop(key)

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

    def train_processing(self, item):
        self.filter_bad_objects(item)
        points, boxes, class_idx = self.augmentation(
            item['points'], item['boxes'], item['class_idx'])
        item.update(dict(points=points, boxes=boxes, class_idx=class_idx))
        self.filter_out_of_bounds(item)
        item['points'] = np.float32(item['points'])
        item['boxes'] = torch.FloatTensor(item['boxes'])
        item['class_idx'] = torch.LongTensor(item['class_idx'])
        self.target_assigner(item)

    def __getitem__(self, idx):
        item = deepcopy(self.annotations[self.inds[idx]])
        item['points'] = read_velo(item['velo_path'])
        if self.split == 'train':
            self.train_processing(item)
        self.drop_keys(item)
        return item
