import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from visdom import Visdom

from pvrcnn.detector import ProposalLoss
from pvrcnn.core import cfg, TrainPreprocessor, VisdomLinePlotter
from pvrcnn.dataset import KittiDataset
from pvrcnn.detector import PV_RCNN


def build_train_dataloader(cfg, preprocessor):
    dataset = KittiDataset(cfg, 'train')
    dataloader = DataLoader(
        dataset,
        collate_fn=preprocessor.collate,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )
    return dataloader


def save_cpkt(model, optimizer, epoch, meta=None):
    fpath = f'./epoch_{epoch}.pth'
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        meta=meta,
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


def update_plot(losses, prefix):
    for key in ['loss', 'cls_loss', 'reg_loss']:
        plotter.update(f'{prefix}_{key}', losses[key].item())


def train_model(model, dataloader, optimizer, loss_fn, epochs, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, epochs):
        for step, item in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            out = model(item, proposals_only=True)
            losses = loss_fn(out)
            losses['loss'].backward()
            optimizer.step()
            if (step % 50) == 0:
                update_plot(losses, 'step')
        update_plot(losses, 'epoch')
        save_cpkt(model, optimizer, epoch)


def get_proposal_parameters(model):
    for p in model.roi_grid_pool.parameters():
        p.requires_grad = False
    for p in model.refinement_layer.parameters():
        p.requires_grad = False
    return model.parameters()


def main():
    model = PV_RCNN(cfg).cuda()
    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    dataloader_train = build_train_dataloader(cfg, preprocessor)
    parameters = get_proposal_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LR)
    start_epoch = load_ckpt('./epoch_0.pth', model, optimizer)
    train_model(model, dataloader_train, optimizer, loss_fn, cfg.TRAIN.EPOCHS, start_epoch)


if __name__ == '__main__':
    global plotter
    plotter = VisdomLinePlotter(env='training')
    main()
