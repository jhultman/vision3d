import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from pvrcnn.detector import ProposalLoss, PV_RCNN
from pvrcnn.core import cfg, TrainPreprocessor, VisdomLinePlotter
from pvrcnn.dataset import KittiDataset


def build_train_dataloader(cfg, preprocessor):
    dataset = KittiDataset(cfg, 'train')
    dataloader = DataLoader(
        dataset,
        collate_fn=preprocessor.collate,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=2,
    )
    return dataloader


def save_cpkt(model, optimizer, epoch, meta=None):
    fpath = f'./ckpts/epoch_{epoch}.pth'
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        meta=meta,
    )
    os.makedirs('./ckpts', exist_ok=True)
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


def to_device(item):
    keys = ['G_cls', 'G_reg', 'M_cls', 'M_reg', 'points',
        'features', 'coordinates', 'occupancy']
    for key in keys:
        item[key] = item[key].cuda()


def train_model(model, dataloader, optimizer, lr_scheduler, loss_fn, epochs, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, epochs):
        for step, item in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            to_device(item)
            optimizer.zero_grad()
            out = model(item, proposals_only=True)
            losses = loss_fn(out)
            losses['loss'].backward()
            optimizer.step()
            if False:
                lr_scheduler.step()
            if (step % 50) == 0:
                update_plot(losses, 'step')
        save_cpkt(model, optimizer, epoch)


def get_proposal_parameters(model):
    for p in model.roi_grid_pool.parameters():
        p.requires_grad = False
    for p in model.refinement_layer.parameters():
        p.requires_grad = False
    return model.parameters()


def main():
    """TODO: Trainer class to manage objects."""
    model = PV_RCNN(cfg).cuda()
    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    dataloader = build_train_dataloader(cfg, preprocessor)
    parameters = get_proposal_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, steps_per_epoch=len(dataloader), epochs=cfg.TRAIN.EPOCHS)
    start_epoch = load_ckpt('./ckpts/epoch_8.pth', model, optimizer)
    train_model(model, dataloader, optimizer, scheduler, loss_fn, cfg.TRAIN.EPOCHS, start_epoch)


from multiprocessing import set_start_method

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    global plotter
    plotter = VisdomLinePlotter(env='training')
    cfg.merge_from_file('../configs/car_lite.yaml')
    main()
