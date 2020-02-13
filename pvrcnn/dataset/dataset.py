from tqdm import tqdm
import pickle
import numpy as np
from copy import deepcopy
from pvrcnn.dataset.kitti import read_calib, read_label, read_velo
import os.path as osp

from torch.utils.data import Dataset


class KittiDataset(Dataset):

    def __init__(self, cfg, split):
        self.split = split
        self.rootdir = cfg.DATA.ROOTDIR
        self.load_annotations(cfg)
        self.cfg = cfg

    def read_splitfile(self, cfg):
        fpath = osp.join(cfg.DATA.SPLITDIR, f'{self.split}.txt')
        self.inds = np.loadtxt(fpath, dtype=np.int32).tolist()

    def try_read_cached_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        if not osp.isfile(fpath):
            return False
        print('Reading cached annotations...')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        print('Done loading annotations.')
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
        print('Done loading annotations.')
        self.cache_annotations(cfg)

    def make_simple_object(self, obj, calib):
        xyz = calib.R0 @ obj.t
        xyz = calib.C2V @ np.r_[xyz, 1]
        wlh = np.r_[obj.w, obj.l, obj.h]
        rz = np.r_[-obj.ry]
        box = np.r_[xyz, wlh, rz]
        obj = dict(box=box,  cls_id=obj.cls_id)
        return obj

    def filter_bad_boxes(self, item):
        boxes, class_ids = [], []
        for obj in item['objects']:
            if (obj['box'][3:6] <= 0).any():
                continue
            if (obj['box'][0:3] <= self.cfg.GRID_BOUNDS[0:3]).any():
                continue
            if (obj['box'][0:3] >= self.cfg.GRID_BOUNDS[3:6]).any():
                continue
            if obj['cls_id'] == -1:
                continue
            boxes += [obj['box']]
            class_ids += [obj['cls_id']]
        item['boxes'] = np.stack(boxes, axis=0)
        item['class_ids'] = np.r_[class_ids]
        return item

    def __getitem__(self, idx):
        idx = self.inds[idx]
        item = deepcopy(self.annotations[idx])
        item['points'] = read_velo(item['velo_path'])
        item['objects'] = [self.make_simple_object(
            obj, item['calib']) for obj in item['objects']]
        if self.split == 'train':
            item = self.filter_bad_boxes(item)
        [item.pop(key) for key in ['velo_path', 'objects', 'calib']]
        return item
