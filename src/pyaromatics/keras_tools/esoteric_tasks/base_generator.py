import tensorflow as tf
import numpy as np

from pyaromatics.stay_organized.utils import str2val


class BaseGenerator(tf.keras.utils.Sequence):
    config = ''
    output_type = '[io]'
    epoch = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch == int(self.epochs / 3) and 'repetitionsschedule' in self.config:
            self.repetitions = int(self.repetitions - 1)

    def __getitem__(self, index=0):
        batch = self.data_generation()
        i, m, o = batch['input_spikes'], batch['mask'], batch['target_output']
        i = np.repeat(i, self.repetitions, axis=1)
        o = np.repeat(o, self.repetitions, axis=1)
        if 'mlminputs' in self.config:
            mlmeps = str2val(self.config, 'mlmeps', int, default=5)

            if self.epoch < mlmeps:

                mask = np.random.choice([0, 1], size=i.shape, p=[0.95, 0.05])
                random_values = np.random.choice(self.vocab_size, size=i.shape)
                i = i * (1 - mask) + random_values * mask

        if self.output_type == '[im]o':
            return (i, m), o
        elif self.output_type == '[imo]':
            return (i, m, o),
        if self.output_type == '[io]':
            return (i, o),
        else:
            raise NotImplementedError
