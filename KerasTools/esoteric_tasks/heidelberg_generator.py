import os, h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from GenericTools.StayOrganizedTools.download_utils import download_and_unzip
from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

HEIDELBERGDIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'SpikingHeidelbergDigits'))
shd_train_filename = os.path.join(HEIDELBERGDIR, "shd_train.h5")
ssc_train_filename = os.path.join(HEIDELBERGDIR, "ssc_train.h5")

if not os.path.isdir(HEIDELBERGDIR):
    os.mkdir(HEIDELBERGDIR)

data_links = [
    'https://compneuro.net/datasets/hd_audio.tar.gz',
    # 'https://compneuro.net/datasets/md5sums.txt',
    # 'https://compneuro.net/datasets/shd_test.h5.gz',
    'https://compneuro.net/datasets/shd_test.h5.zip',
    # 'https://compneuro.net/datasets/shd_train.h5.gz',
    'https://compneuro.net/datasets/shd_train.h5.zip',
    # 'https://compneuro.net/datasets/ssc_test.h5.gz',
    'https://compneuro.net/datasets/ssc_test.h5.zip',
    # 'https://compneuro.net/datasets/ssc_train.h5.gz',
    'https://compneuro.net/datasets/ssc_train.h5.zip',
    # 'https://compneuro.net/datasets/ssc_valid.h5.gz',
    'https://compneuro.net/datasets/ssc_valid.h5.zip',
]

def test_non_spiking():
    f = h5py.File(ssc_train_filename, "r")

    # List all groups
    print("Keys: %s" % f.keys())
    print("Keys: %s" % f['spikes'].keys())
    a_group_key = list(f.keys())[0]

    units = f['spikes']['units']
    times = f['spikes']['times']
    labels = f['labels']

    print(units[0])
    print(times[0])

def test_spiking(plot_it=True):
    f = h5py.File(shd_train_filename, "r")

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    units = f['spikes']['units']
    times = f['spikes']['times']
    labels = f['labels']

    # A quick raster plot for one of the samples
    if plot_it:
        fig = plt.figure(figsize=(16, 4))
        idx = np.random.randint(len(times), size=3)

        bins = np.linspace(0, 1, 250)
        for i, k in enumerate(idx):
            ax = plt.subplot(1, 3, i + 1)

            dense = np.zeros((700, 250))
            tr = times[k]
            which = tr < 1
            tr = tr[which]
            u = 699 - units[k][which]
            binned = np.digitize(tr, bins, right=True)  # times[k]
            dense[u, binned] = 1
            ax.pcolormesh(dense)
            # ax.scatter(binned, 700 - u, color="k", alpha=0.33, s=2)
            ax.set_title("Label %i" % labels[k])

        plt.show()
        f.close()


class SpokenHeidelbergDigits(BaseGenerator):

    def __init__(
            self,
            epochs=1,
            tvt='train',
            batch_size=32,
            n_dt_per_step=3,
            steps_per_epoch=None,
            download_if_missing=False):

        if download_if_missing:
            download_and_unzip(data_links, HEIDELBERGDIR)

        self.__dict__.update(batch_size=batch_size, tvt=tvt,
                             steps_per_epoch=steps_per_epoch, n_dt_per_step=n_dt_per_step)

        self.in_dim = 700
        self.out_dim = 20
        self.in_len = 250 * n_dt_per_step
        self.out_len = 250
        self.epochs = 300 if epochs == None else epochs

        self.batch_size = batch_size
        self.batch_index = 0

        self.on_epoch_end()
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size)) \
            if steps_per_epoch == None else steps_per_epoch

    def on_epoch_end(self):
        self.batch_index = 0

        if self.tvt == 'test':
            self.f = h5py.File(shd_train_filename.replace('train', 'test'), "r")
            set = range(0, 2088)

        elif self.tvt in ['validation', 'val']:
            self.f = h5py.File(shd_train_filename, "r")
            set = range(int(9 * 8155 / 10), 8155)

        elif self.tvt == 'train':
            self.f = h5py.File(shd_train_filename, "r")
            set = range(0, int(9 * 8155 / 10))

        else:
            raise NotImplementedError

        self.nb_lines = set[-1] - set[0]

        self.units = self.f['spikes']['units'][()][set]
        self.times = self.f['spikes']['times'][()][set]
        self.labels = self.f['labels'][()][set]

    def data_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size

        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        batch = []
        target = []
        bins = np.linspace(0, 1, self.in_len)
        for i in range(batch_start, batch_stop):
            dense = np.zeros((self.in_dim, self.in_len))
            tr = self.times[i]
            which = tr < 1
            tr = tr[which]
            u = self.in_dim - 1 - self.units[i][which]
            binned = np.digitize(tr, bins, right=True)  # times[k]
            dense[u, binned] = 1
            batch.append(dense.T[None])
            target.append(self.labels[i][None])

        batch = np.concatenate(batch, 0)
        target = np.concatenate(target, 0)

        new_mask = np.ones((self.batch_size, self.out_len, self.out_dim))
        target_oh = tf.keras.utils.to_categorical(target, num_classes=self.out_dim)[:, None, :]

        target_oh = np.repeat(target_oh, self.out_len, 1)

        return {'input_spikes': batch, 'target_output': target_oh,
                'mask': new_mask}


if __name__ == '__main__':
    download_and_unzip(data_links, HEIDELBERGDIR)
    test_spiking()
    test_non_spiking()
