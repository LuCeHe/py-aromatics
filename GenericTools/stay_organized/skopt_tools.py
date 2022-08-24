import numpy as np
from tqdm import tqdm

class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)
        self.min = np.inf
        self.params = []

    def __call__(self, res):
        if res.fun < self.min:
            self.min = res.fun
            self.params = res.x
        self._bar.update()
        self._bar.set_postfix({'Loss': self.min, 'Params': self.params})
