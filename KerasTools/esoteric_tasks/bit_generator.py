


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


class OneBitTimeDependentGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            batch_size=32,
            steps_per_epoch=1,
            timesteps_dependency=3,
            maxlen=20,
            neutral_phases=True,
            repetitions=2):
        'Initialization'
        self.__dict__.update(steps_per_epoch=steps_per_epoch,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             timesteps_dependency=timesteps_dependency,
                             neutral_phases=neutral_phases,
                             repetitions=repetitions)

        self.transition_matrix = np.random.rand(*((2,) * timesteps_dependency))

        self.in_dim = 1
        self.out_dim = 2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        return batch

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        initial_sequences = np.random.choice(2, (self.batch_size, self.timesteps_dependency))
        probabilities = np.zeros_like(initial_sequences)

        sequences = initial_sequences

        if self.neutral_phases:
            maxlen = int(self.maxlen / 2 / self.repetitions)
        else:
            maxlen = self.maxlen

        for _ in range(maxlen - self.timesteps_dependency + 1):
            last_timesteps = sequences[:, -self.timesteps_dependency:]
            next_probs = np.array([self.transition_matrix[tuple(l)] for l in tuple(last_timesteps)])
            next_symbol = 1 * (next_probs > np.random.rand(self.batch_size))
            sequences = np.c_[sequences, next_symbol]
            probabilities = np.c_[probabilities, next_probs]

        if self.neutral_phases:
            add_neutral_phases = np.zeros((self.batch_size, (maxlen + 1) * 2), dtype=np.int32)
            add_neutral_phases[:, ::2] = sequences
            sequences = add_neutral_phases

            add_neutral_phases = np.zeros((self.batch_size, (maxlen + 1) * 2), dtype=np.float32)
            add_neutral_phases[:, ::2] = probabilities
            probabilities = add_neutral_phases

        sequences = np.array(sequences, dtype='float32')
        x = sequences[:, :-2, np.newaxis]
        y = sequences[:, 1:-1, np.newaxis]
        prob_x = probabilities[:, :-2, np.newaxis]
        prob_y = probabilities[:, 1:-1, np.newaxis]

        # repeat
        x, y, prob_x, prob_y = [np.repeat(t, self.repetitions, axis=1)[:, :self.maxlen] for t in [x, y, prob_x, prob_y]]
        y = tf.keras.utils.to_categorical(y, num_classes=2)
        mask = np.ones_like(x[..., 0])

        return {'input_spikes': x[:, :self.maxlen, :], 'output_class': y[:, :self.maxlen, :],
                'output_mask': mask[:, :self.maxlen], 'prob_x': prob_x, 'prob_y': prob_y}


if __name__ == '__main__':
    batch_size = 3
    generator = OneBitTimeDependentGenerator(
        batch_size=batch_size,
        steps_per_epoch=1,
        timesteps_dependency=3,
        maxlen=30,
        neutral_phases=True)

    batch = generator.__getitem__()
    print(batch[0].shape)
    print(batch[1].shape)
    print()
    print(batch[0])
    print()
    print(batch[1])
