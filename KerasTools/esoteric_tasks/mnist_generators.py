import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator


def generate_poisson_noise_np(prob_pattern, n_neurons=1, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    if not (freezing_seed is None):
        rng = rd.RandomState(freezing_seed)
    else:
        rng = rd.RandomState()

    # repeat as a number of different neurons
    prob_pattern = np.repeat(prob_pattern, n_neurons, axis=2)
    shp = prob_pattern.shape

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return 1 * spikes


def image2spikes(n_input, input_px, n_dt_per_step):
    current_batch_size = input_px.shape[0]
    waiting_and_reply_period = np.zeros((current_batch_size, (100 + 56) * n_dt_per_step, n_input - 1))
    input_px = np.expand_dims(input_px, axis=2) / 256
    input_px = np.repeat(input_px, n_dt_per_step, 1)
    spikes = generate_poisson_noise_np(input_px, n_neurons=n_input - 1)

    spikes_and_wait = np.concatenate([spikes, waiting_and_reply_period], axis=1)

    # plus a signal neuron that tells the net when to give a reply
    signal_neuron_zeros = np.zeros((current_batch_size, spikes_and_wait.shape[1] - 56 * n_dt_per_step, 1))
    signal_neuron_ones = np.ones((current_batch_size, 56 * n_dt_per_step, 1))
    signal_neuron = np.concatenate([signal_neuron_zeros, signal_neuron_ones], axis=1)

    digit_code = np.concatenate([signal_neuron, spikes_and_wait], axis=2)
    return digit_code


class SeqMNIST(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            tvt='train',
            batch_size=32,
            keep=1.,
            num_input=1,
            n_dt_per_step=3,
            steps_per_epoch=None,
            permuted=True,
            inherit_from_gen=False):

        self.__dict__.update(batch_size=batch_size, tvt=tvt, keep=keep, permuted=permuted,
                             num_input=num_input, steps_per_epoch=steps_per_epoch, n_dt_per_step=n_dt_per_step)

        self.create_dataset()

        self.in_dim = num_input
        self.out_dim = self.num_classes
        self.in_len = self.length * n_dt_per_step
        self.out_len = self.length
        self.epochs = 300 if epochs == None else epochs  # int(36000/30000*batch_size) train in 2 rounds to get to 614 epochs

        if self.permuted:
            if inherit_from_gen is False:
                self.p = np.random.permutation(self.width * self.height)
            else:
                self.p = inherit_from_gen.p
                del inherit_from_gen

        else:
            self.p = np.arange(self.width * self.height)

        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(len(self.x) / self.batch_size)

    def create_dataset(self):
        self.num_classes = 10
        train_split = .8

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        len_train_val = len(x_train)

        self.width, self.height = x_train.shape[1], x_train.shape[2]

        if self.tvt == 'test':
            x = x_test
            y = y_test
        elif self.tvt in ['validation', 'val']:
            x = x_train[int(train_split * len_train_val):]
            y = y_train[int(train_split * len_train_val):]
        elif self.tvt == 'train':
            x = x_train[:int(train_split * len_train_val)]
            y = y_train[:int(train_split * len_train_val)]
        else:
            raise NotImplementedError

        self.x = x.reshape((-1, self.width * self.height))
        self.y = tf.keras.utils.to_categorical(y, self.num_classes)

        self.length = self.width * self.height  # + 100 + 56

    def on_epoch_end(self):
        self.create_dataset()

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        random_idx = np.random.choice(len(self.x), self.batch_size)
        input_px = self.x[random_idx][:, self.p]
        target_oh = self.y[random_idx]
        # batch = image2spikes(self.num_input, input_px, self.n_dt_per_step)

        # FIXME: probably reconstruction is already fairly difficult for \
        # LSNN, but upgrade it to next timestep prediction

        new_mask = np.ones((self.batch_size, self.length, self.num_classes))
        input_px = np.repeat(input_px, self.n_dt_per_step, 1)[..., None]
        target_oh = np.repeat(target_oh[:, None, :], self.length, 1)

        return {'input_spikes': input_px, 'target_output': target_oh, 'mask': new_mask}


def download():
    mnist.load_data()


if __name__ == '__main__':
    download()
