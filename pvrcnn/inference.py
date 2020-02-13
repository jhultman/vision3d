import numpy as np

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN


def make_points(n, cfg):
    points = np.random.uniform(
        0, 50, size=(n, cfg.C_IN)).astype(np.float32)
    return points


def main():
    preprocessor = Preprocessor(cfg)
    net = PV_RCNN(cfg, preprocessor).cuda()
    points = [make_points(95000, cfg), make_points(90000, cfg)]
    out = net(points)


if __name__ == '__main__':
    main()
