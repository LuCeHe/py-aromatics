import os
import numpy as np

import tensorflow as tf
from pyaromatics.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from sg_design_lif.generate_data.lca_utils import norm_spikegram

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'lca_hd_700'))


class LCAGenerator(NumpyClassificationGenerator):

    def __init__(self,
                 epochs=1,
                 tvt='train',
                 batch_size=32,
                 steps_per_epoch=None,
                 config=''
                 ):

        if tvt == 'train':
            input_path = os.path.join(DATAPATH, 'train_spikegram.npy')
            output_path = os.path.join(DATAPATH, 'train_labels.npy')

            X = np.load(input_path)
            y = np.load(output_path)

            X = np.reshape(X, (4011, -1, 700))
            val_split = .1
            X, y = X[:-int(val_split * len(X))], y[:-int(val_split * len(X))]

        elif tvt in ['val', 'validation']:

            input_path = os.path.join(DATAPATH, 'train_spikegram.npy')
            output_path = os.path.join(DATAPATH, 'train_labels.npy')

            X = np.load(input_path)
            y = np.load(output_path)

            X = np.reshape(X, (4011, -1, 700))
            val_split = .1
            X, y = X[-int(val_split * len(X)):], y[-int(val_split * len(X)):]

        elif tvt == 'test':
            input_path = os.path.join(DATAPATH, 'test_spikegram.npy')
            output_path = os.path.join(DATAPATH, 'test_labels.npy')

            X = np.load(input_path)
            y = np.load(output_path)

            X = np.reshape(X, (1079, -1, 700))

        self.length = X.shape[1]
        self.in_dim = X.shape[2]
        self.out_dim = 20
        self.in_len = self.length
        self.out_len = self.length
        self.config=config

        if 'spikelca' in config:
            X = tf.sign(X).numpy()

        y = np.repeat(y[..., None], self.length, 1)

        self.epochs = 450 if epochs == None else epochs
        self.batch_size = batch_size

        super().__init__(
            X=X, y=y,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    train_spikegram_path = os.path.join(DATAPATH, 'train_spikegram.npy')
    # test_spikegram_path = os.path.join(DATAPATH, 'test_spikegram.npy')

    if not os.path.exists(train_spikegram_path):
        for set in ['train', 'test']:
            input_path = os.path.join(DATAPATH, f'{set}_set.npy')
            X = np.load(input_path)
            spg = norm_spikegram(X)
            path = os.path.join(DATAPATH, f'{set}_spikegram.npy')
            np.save(path, spg)

    # import tensorflow as tf
    # path = os.path.join(DATAPATH, f'test_spikegram.npy')
    # X = np.load(path)
    # spikes = tf.sign(X)
    # print('n spikes: ', tf.math.count_nonzero(spikes).numpy()/spikes.shape[0]/spikes.shape[1])