"""
minor changes on
https://github.com/WittmannF/LRFinder
"""

import os
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class LRFinder(Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5, path_tmp=''):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        self.path_tmp = os.path.join(path_tmp, 'tmp.hdf5')
        self.path_plot = os.path.join(path_tmp, 'loss_vs_lr.png')
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs'] * p['samples'] // p['batch_size']
        except:
            n_iterations = p['steps'] * p['epochs']

        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations // self.batches_lr_update + 1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        self.best_lr = 0

        if self.reload_weights:
            self.model.save_weights(self.path_tmp)

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss
            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            self.best_lr = lr

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights(self.path_tmp)

            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights(self.path_tmp)

        os.remove(self.path_tmp)
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.title('min lr {:f}'.format(self.best_lr))
        # plt.vlines(self.best_lr, min(self.losses), max(self.losses))
        # plt.text(self.best_lr, 0, '{} lr'.format(self.best_lr), rotation=90)

        # plt.show()

        plt.savefig(self.path_plot, bbox_inches='tight')
        plt.close('all')
