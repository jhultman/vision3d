import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from pvrcnn.detector import ProposalLoss
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
    return dict(batch_item)


def build_train_dataloader(cfg):
    dataset = KittiDataset(cfg, 'train')
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )
    return dataloader


def save_cpkt(model, optimizer, epoch):
    fpath = f'./epoch_{epoch}.pth'
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
    )
    torch.save(ckpt, fpath)


def load_ckpt(fpath, model, optimizer):
    if not osp.isfile(fpath):
        return 0
    ckpt = torch.load(fpath)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    return epoch


def train_model(model, dataloader, optimizer, loss_fn, epochs, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, epochs):
        for item in tqdm(dataloader):
            out = model(item, proposals_only=True)
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()
        save_cpkt(model, optimizer, epoch)


def get_proposal_parameters(model):
    for p in model.roi_grid_pool.parameters():
        p.requires_grad = False
    for p in model.refinement_layer.parameters():
        p.requires_grad = False
    return model.parameters()


def main():
    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    model = PV_RCNN(cfg, preprocessor).cuda()
    dataloader_train = build_train_dataloader(cfg)
    parameters = get_proposal_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LR)
    start_epoch = load_ckpt('./epoch_0.pth', model, optimizer)
    train_model(model, dataloader_train, optimizer, loss_fn, cfg.TRAIN.EPOCHS, start_epoch)


if __name__ == '__main__':
    main()
