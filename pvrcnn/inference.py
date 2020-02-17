import numpy as np
import torch

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN


def make_points(n, cfg):
    lower, upper = np.split(cfg.GRID_BOUNDS, [3])
    points = np.random.rand(n, 3) * (upper - lower) + lower
    points = np.pad(points, ((0, 0), (0, 1))).astype(np.float32)
    return points


def main():
    preprocessor = Preprocessor(cfg)
    net = PV_RCNN(cfg).cuda().eval()
    item = dict(points=[make_points(95000, cfg), make_points(90000, cfg)])
    with torch.no_grad():
        item = preprocessor(item)
        out = net(item, proposals_only=True)


if __name__ == '__main__':
    main()
