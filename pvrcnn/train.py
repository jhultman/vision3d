import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from pvrcnn.core import cfg, TrainPreprocessor
from pvrcnn.dataset import KittiDataset
from pvrcnn.detector import PV_RCNN


def to_cuda(item):
    item['class_ids'] = torch.from_numpy(item['class_ids'].astype(np.int64)).cuda()
    item['boxes'] = torch.from_numpy(item['boxes'].astype(np.float32)).cuda()
    return item


def collate_fn(items):
    batch_item = defaultdict(list)
    for item in items:
        for key, val in to_cuda(item).items():
            batch_item[key] += [val]
    return batch_item


def build_train_dataloader(cfg):
    dataset = KittiDataset(cfg, 'train')
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )
    return dataloader


def train_model(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for item in dataloader:
            out = model(item)
            break
        break


def main():
    preprocessor = TrainPreprocessor(cfg)
    model = PV_RCNN(cfg, preprocessor).cuda()
    dataloader_train = build_train_dataloader(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    train_model(model, dataloader_train, optimizer, epochs=cfg.TRAIN.EPOCHS)


if __name__ == '__main__':
    main()
