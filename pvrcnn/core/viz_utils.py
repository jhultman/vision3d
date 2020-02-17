import numpy as np
from collections import defaultdict
from visdom import Visdom


class AverageMeter:

    def __init__(self):
        self.totals = defaultdict(float)
        self.tallies = defaultdict(int)
        self.averages = defaultdict(float)

    def _update(self, key, val):
        self.tallies[key] += 1
        self.totals[key] += val
        self.averages[key] = self.totals[key] / self.tallies[key]

    def __getitem__(self, key):
        return self.averages[key]


class VisdomLinePlotter:

    def __init__(self, env='main'):
        self.viz = Visdom()
        self.env = env
        self.meter = AverageMeter()
        self.windows = defaultdict(lambda: None)

    def get_window_update(self, key):
        win = self.windows[key]
        update = 'append' if win else None
        return win, update

    def get_kwargs(self, key):
        win, update = self.get_window_update(key)
        opts = dict(xlabel='steps', ylabel=key, title=key.capitalize())
        kwargs = dict(name=key, update=update, env=self.env, win=win, opts=opts)
        return kwargs

    def update(self, key, val):
        self.meter._update(key, val)
        y = np.r_[self.meter[key]]
        x = np.r_[self.meter.tallies[key]]
        kwargs = self.get_kwargs(key)
        self.windows[key] = self.viz.line(y, x, **kwargs)
