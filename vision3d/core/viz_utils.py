import numpy as np
from collections import defaultdict
from visdom import Visdom


class AverageMeter:

    def __init__(self):
        self.total = defaultdict(float)
        self.tally = defaultdict(int)
        self.current = defaultdict(float)
        self.average = defaultdict(float)

    def _update(self, key, val):
        self.tally[key] += 1
        self.total[key] += val
        self.current[key] = val
        self.average[key] = self.total[key] / self.tally[key]


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
        x = np.r_[self.meter.tally[key]]
        metrics = [self.meter.average, self.meter.current]
        keys = [key + '_avg', key + '_cur']
        for metric, _key in zip(metrics, keys):
            y = np.r_[metric[key]]
            kwargs = self.get_kwargs(_key)
            self.windows[_key] = self.viz.line(y, x, **kwargs)
