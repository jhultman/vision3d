import numpy as np
import torch

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN


def make_points(n, cfg):
    points = np.random.uniform(
        0, 50, size=(n, cfg.C_IN)).astype(np.float32)
    return points


def main():
    preprocessor = Preprocessor(cfg)
    net = PV_RCNN(cfg, preprocessor).cuda().eval()
    with torch.no_grad():
        input_dict = dict(points=[make_points(95000, cfg), make_points(90000, cfg)])
        out = net(input_dict, proposals_only=True)


if __name__ == '__main__':
    main()
