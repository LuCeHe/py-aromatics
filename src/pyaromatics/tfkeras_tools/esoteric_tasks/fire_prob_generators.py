import numpy as np
import tensorflow as tf


class BaseGenerator_1(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            batch_size=32,
            steps_per_epoch=100,
            time_steps=200,
            max_firing=.8,
            min_firing=.2,
            n_in=1,
            n_out=3,
            quantity_to_predict='phase'):

        self.__dict__.update(batch_size=batch_size,
                             steps_per_epoch=steps_per_epoch,
                             time_steps=time_steps,
                             max_firing=max_firing,
                             min_firing=min_firing,
                             n_in=n_in,
                             n_out=n_out,
                             quantity_to_predict=quantity_to_predict
                             )
        self.half_time = int(self.time_steps / 2)

        self.create_dataset()

    def create_dataset(self):
        # produce n sinusoidal prob distributions for different inputs
        self.input_amplitudes = (self.max_firing - self.min_firing) * np.random.rand(self.n_in) + self.min_firing
        self.input_frequencies = np.random.rand(self.n_in)
        self.input_phase = 10 * np.random.rand(self.n_in)

        # produce n sinusoidal prob distributions for different outputs, function of the input
        self.output_amplitudes = .7
        self.output_frequencies = .5
        self.output_phase = 0

        # sum of two inputs shifted in time
        if self.quantity_to_predict == 'phase':

            shuffle_1 = np.random.choice(self.input_phase, self.n_out)
            shuffle_2 = np.random.choice(self.input_phase, self.n_out)
            self.output_phase = shuffle_1 + shuffle_2
        else:
            raise NotImplementedError

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        X, y = self.data_generation()
        return X, y

    def on_epoch_end(self):
        pass

    def prob2spike(self, input):
        spikes = input > np.random.rand(*input.shape)
        return spikes.astype(np.float64)

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # input
        x = np.linspace(0, 100, self.time_steps)[np.newaxis, :, np.newaxis]
        x = np.repeat(x, self.n_in, axis=2)
        x = np.repeat(x, self.batch_size, axis=0)

        input = self.input_amplitudes * np.sin(self.input_frequencies * x + self.input_phase)

        # output
        # x shifted by one to do one time step prediction
        x = np.linspace(1, 101, self.time_steps)[np.newaxis, :, np.newaxis]
        x = np.repeat(x, self.n_out, axis=2)
        x = np.repeat(x, self.batch_size, axis=0)

        output = self.output_amplitudes * np.sin(self.output_frequencies * x + self.output_phase)
        zeros = np.zeros((self.half_time, self.n_out))
        output[:, :self.half_time, :] = zeros

        # probability to spike
        input_spikes = self.prob2spike(input)
        output_spikes = self.prob2spike(output)
        return input_spikes, output_spikes


class BaseGenerator_2(BaseGenerator_1):
    def prob2spike(self, input):
        spikes = input > .45
        return spikes.astype(np.float64)


class AeGenerator(BaseGenerator_2):
    pass


class VaeGenerator(BaseGenerator_2):

    def __getitem__(self, index=0):
        X, y = self.data_generation()
        return X, [y, y]
