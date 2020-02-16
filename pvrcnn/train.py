import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from visdom import Visdom

from pvrcnn.detector import ProposalLoss
from pvrcnn.core import cfg, TrainPreprocessor
from pvrcnn.dataset import KittiDataset
from pvrcnn.detector import PV_RCNN


class VisdomLinePlotter:

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def new_plot(self, var_name, split_name, title_name, xlabel, x, y):
        opts = dict(legend=[split_name], title=title_name, xlabel=xlabel, ylabel=var_name)
        self.plots[var_name] = self.viz.line(
            X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=opts)

    def plot(self, var_name, split_name, title_name, xlabel, x, y):
        if var_name not in self.plots:
            self.new_plot(var_name, split_name, title_name, xlabel, x, y)
        else:
            self.viz.line(
                X=np.array([x]), Y=np.array([y]), env=self.env,
                win=self.plots[var_name], name=split_name, update='append')


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


def train_model(model, dataloader, optimizer, loss_fn, epochs, start_epoch=0):
    model.train()
    total_step = 0
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for step, item in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            out = model(item, proposals_only=True)
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_step += 1
            if (step % 100) == 0:
                plotter.plot('step_loss', 'train', 'Step Loss', 'step', total_step, total_loss / (step + 1))
        plotter.plot('epoch_loss', 'train', 'Epoch Loss', 'epoch', epoch, total_loss / (step + 1))
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
    global plotter
    plotter = VisdomLinePlotter(env_name='training')
    main()
