import os

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from GenericTools.keras_tools.esoteric_tasks.mnist import getMNIST
from GenericTools.keras_tools.esoteric_tasks.base_generator import BaseGenerator

# CDIR = os.path.dirname(os.path.realpath(__file__))
# DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'mnist'))
# if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)


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
            data_dir,
            epochs=1,
            tvt='train',
            batch_size=32,
            keep=1.,
            num_input=1,
            repetitions=3,
            steps_per_epoch=None,
            permuted=False,
            original_size=True,
            spike_latency=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.data_dir = os.path.join(data_dir, 'mnist')
        os.makedirs(self.data_dir, exist_ok=True)
        self.__dict__.update(batch_size=batch_size, tvt=tvt, keep=keep, permuted=permuted, original_size=original_size,
                             num_input=num_input, steps_per_epoch=steps_per_epoch, repetitions=repetitions,
                             spike_latency=spike_latency)

        self.create_dataset()

        self.in_dim = num_input if not spike_latency else 28 * 28
        self.out_dim = self.num_classes
        self.in_len = self.length * repetitions
        self.out_len = self.length * repetitions
        self.epochs = 300 if epochs == None else epochs  # int(36000/30000*batch_size) train in 2 rounds to get to 614 epochs

        if self.permuted:
            permutation_file = os.path.join(data_dir, 'permutation_size{}.npy'.format(self.original_size))
            if not os.path.isfile(permutation_file):
                self.p = np.random.permutation(self.length)
                np.save(permutation_file, self.p)
            else:
                self.p = np.load(permutation_file)

        else:
            self.p = np.arange(self.length)

        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(len(self.x) / self.batch_size)

    def create_dataset(self):
        self.num_classes = 10
        path = os.path.join(self.data_dir, 'mnist.npz')
        self.x, self.y = getMNIST(categorical=False, sequential=True,
                                  original_size=self.original_size, data_split=self.tvt,
                                  spike_latency=self.spike_latency, path=path)
        self.length = self.x.shape[1] if not self.spike_latency else 100

    def on_epoch_end(self):
        self.create_dataset()

    def data_generation(self):
        random_idx = np.random.choice(len(self.x), self.batch_size)
        input_px = self.x[random_idx]
        target_oh = self.y[random_idx]
        # batch = image2spikes(self.num_input, input_px, self.n_dt_per_step)

        if self.spike_latency:
            max_latency = self.length
            too_high = input_px > max_latency
            input_px[too_high] = -1
            categorical_labels = tf.keras.utils.to_categorical(input_px, num_classes=max_latency)

            neg = (input_px > 0)[..., None]
            input_px = categorical_labels * neg
            s = categorical_labels.shape
            input_px = input_px.reshape((-1, s[2], s[1]))
        else:
            input_px = input_px[..., None]

        # input_px = np.repeat(input_px, self.n_dt_per_step, 1)[..., None]
        target_oh = np.repeat(target_oh[:, None], self.length, 1)
        # return {'input_spikes': input_px.astype('float32')[:, self.p], 'target_output': target_oh, 'mask': 1}
        return {'input_spikes': input_px.astype('float32')[:, self.p], 'target_output': target_oh, 'mask': 1}


def download():
    mnist.load_data()


def plot_mnist_sl():
    import matplotlib.pyplot as plt
    x, _ = getMNIST(categorical=False, sequential=False, original_size=True, data_split='train', spike_latency=False)
    # xs, _ = getMNIST(categorical=False, sequential=True, original_size=True, training_set='train', spike_latency=True)

    gen = SeqMNIST(repetitions=1, original_size=True, spike_latency=True)

    batch = gen.__getitem__()

    fig, axs = plt.subplots(2, 1, gridspec_kw={'wspace': .0, 'hspace': .5}, figsize=(4, 4))

    idx = np.random.randint(len(x))
    print(idx)
    idx = 9954

    image = x[idx, ..., 0].T
    image = np.rot90(image, 1, (0, 1))
    raster = batch[0][0][2].T  # xs[idx]
    indices = np.argwhere(raster == 1)
    print(indices.shape)

    print(image.shape, raster.shape)
    axs[0].pcolormesh(image, cmap='Greys')
    # axs[1].pcolormesh(raster, cmap='Greys')
    # axs[1].plot(indices[:, 1], indices[:, 0])
    plt.scatter(indices[:, 1], indices[:, 0], s=.2, alpha=1, c='k')

    # ax.set_title(str(f['extra']['keys'][labels[k]].decode('utf-8')))

    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_yticks([0, 784])
    axs[1].set_xlabel('time (ms)')
    axs[1].set_ylabel('channel')

    plt.savefig('slmnist.png', bbox_inches='tight', dpi=500, transparent=True)
    plt.show()


if __name__ == '__main__':
    # download()
    plot_mnist_sl()
