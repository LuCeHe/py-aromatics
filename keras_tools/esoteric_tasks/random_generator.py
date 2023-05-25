


import numpy as np
import tensorflow as tf
import itertools as itt

from itertools import combinations
from functools import reduce  # not necessary in python 2.x


def k_bits_on(k, n):
    one_at = lambda v, i: v[:i] + [1] + v[i + 1:]
    return [tuple(reduce(one_at, c, [0] * n)) for c in combinations(range(n), k)]


def generate_time_dependency_matrix(timesteps_back=3):
    m = itt.permutations([1, 0, 0])
    return list(m)


class RandomGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20, ):
        'Initialization'
        self.__dict__.update(steps_per_epoch=steps_per_epoch,
                             batch_size=batch_size,
                             maxlen=maxlen)

        self.in_dim = 3
        self.out_dim = 2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        return batch

    def data_generation(self):
        x = np.random.choice(2, [self.batch_size, self.maxlen, self.in_dim])
        y = np.random.choice(2, [self.batch_size, self.maxlen, self.out_dim])
        mask = np.random.choice([True], [self.batch_size, self.maxlen])

        return {'input_spikes': x[:, :self.maxlen, :], 'output_class': y[:, :self.maxlen, :],
                'output_mask': mask[:, :self.maxlen]}
