import numpy as np
import torch
from torch.utils.data import DataLoader

from pvrcnn.core import cfg
from pvrcnn.dataset import KittiDataset
from pvrcnn.detector import PV_RCNN


def collate_fn(item):
    return item


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
            pass


def main():
    model = PV_RCNN(cfg).cuda()
    dataloader_train = build_train_dataloader(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    train_model(model, dataloader_train, optimizer, epochs=cfg.TRAIN.EPOCHS)


if __name__ == '__main__':
    main()
