import numpy as np
import torch
import matplotlib.pyplot as plt

from vision3d.core import cfg, Preprocessor, AnchorGenerator
from vision3d.core.bev_drawer import Drawer
from vision3d.detector import Second


def viz_detections(points, boxes):
    boxes = boxes.cpu().numpy()
    bev_map = Drawer(points, [boxes]).image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bev_map.transpose(1, 0, 2)[::-1])
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def get_model(cfg):
    cfg.merge_from_file('../configs/second/car.yaml')
    anchors = AnchorGenerator(cfg).anchors
    preprocessor = Preprocessor(cfg)
    model = Second(cfg).cuda().eval()
    ckpt = torch.load('../vision3d/ckpts/epoch_12.pth')['state_dict']
    model.load_state_dict(ckpt, strict=True)
    return model, preprocessor, anchors


def main():
    model, preprocessor, anchors = get_model(cfg)
    fpath = '../data/kitti/training/velodyne_reduced/000032.bin'
    points = np.fromfile(fpath, np.float32).reshape(-1, 4)
    with torch.no_grad():
        item = preprocessor(dict(points=[points], anchors=anchors))
        for key in ['points', 'features', 'coordinates', 'occupancy', 'anchors']:
            item[key] = item[key].cuda()
        boxes, batch_idx, class_idx, scores = model.inference(item)
    viz_detections(points, boxes)


if __name__ == '__main__':
    main()
