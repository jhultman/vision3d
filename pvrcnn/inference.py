import os.path as osp
import numpy as np
import torch

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN


def to_device(item):
    for key in ['points', 'features', 'coordinates', 'occupancy']:
        item[key] = item[key].cuda()
    return item


def main():
    preprocessor = Preprocessor(cfg)
    net = PV_RCNN(cfg).cuda().eval()
    basedir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced/')
    item = dict(points=[
        np.fromfile(osp.join(basedir, '000007.bin'), np.float32).reshape(-1, 4),
        np.fromfile(osp.join(basedir, '000008.bin'), np.float32).reshape(-1, 4),
    ])
    with torch.no_grad():
        item = to_device(preprocessor(item))
        out = net(item, proposals_only=True)


if __name__ == '__main__':
    main()
