from tqdm import tqdm
import pickle
import numpy as np
import torch
from copy import deepcopy
import os.path as osp
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner
from .kitti_utils import read_calib, read_label, read_velo


class KittiDataset(Dataset):

    def __init__(self, cfg, split):
        self.split = split
        self.rootdir = cfg.DATA.ROOTDIR
        self.load_annotations(cfg)
        if split == 'train':
            self.target_assigner = ProposalTargetAssigner(cfg)
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
        print('Reading cached annotations.')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        return True

    def cache_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def create_anno(self, idx, cfg):
        velo_path = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced', f'{idx:06d}.bin')
        calib = read_calib(osp.join(cfg.DATA.ROOTDIR, 'calib', f'{idx:06d}.txt'))
        objects = read_label(osp.join(cfg.DATA.ROOTDIR, 'label_2', f'{idx:06d}.txt'))
        annotation = dict(velo_path=velo_path, calib=calib, objects=objects, idx=idx)
        return annotation

    def load_annotations(self, cfg):
        self.read_splitfile(cfg)
        if self.try_read_cached_annotations(cfg):
            return
        print('Generating annotations...')
        self.annotations = dict()
        for idx in tqdm(self.inds):
            self.annotations[idx] = self.create_anno(idx, cfg)
        self.cache_annotations(cfg)

    def make_simple_object(self, obj, calib):
        """Convert coordinates to velodyne frame."""
        xyz = calib.R0 @ obj.t
        xyz = calib.C2V @ np.r_[xyz, 1]
        wlh = np.r_[obj.w, obj.l, obj.h]
        rz = np.r_[-obj.ry]
        box = np.r_[xyz, wlh, rz]
        obj = dict(box=box, class_id=obj.class_id)
        return obj

    def stack_boxes(self, item):
        boxes = [obj['box'] for obj in item['objects']]
        class_idx = [obj['class_idx'] for obj in item['objects']]
        boxes = torch.FloatTensor(boxes)
        class_idx = torch.LongTensor(class_idx)
        item.update(dict(boxes=boxes, class_idx=class_idx))

    def make_objects(self, item):
        item['objects'] = [self.make_simple_object(
            obj, item['calib']) for obj in item['objects']]
        self.stack_boxes(item)

    def drop_keys(self, item):
        for key in ['velo_path', 'objects', 'calib']:
            item.pop(key)

    def filter_bad_boxes(self, item):
        class_idx = item['class_idx'][:, None]
        xyz, wlh, _ = item['boxes'].split([3, 3, 1], dim=1)
        lower, upper = torch.tensor(self.cfg.GRID_BOUNDS).split([3, 3])
        keep = ((xyz >= lower) & (xyz <= upper) &
            (class_idx != -1) & (wlh > 0)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def __getitem__(self, idx):
        item = deepcopy(self.annotations[self.inds[idx]])
        item['points'] = read_velo(item['velo_path'])
        self.make_objects(item)
        if self.split == 'train':
            self.filter_bad_boxes(item)
            self.target_assigner(item)
        self.drop_keys(item)
        return item
