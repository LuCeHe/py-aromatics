import os, h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from GenericTools.stay_organized.download_utils import download_and_unzip
from GenericTools.keras_tools.esoteric_tasks.base_generator import BaseGenerator
from GenericTools.keras_tools.esoteric_tasks.heidelberg_preprocess import generate_dataset

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', '..', 'data'))

data_links = [
    # 'https://compneuro.net/datasets/hd_audio.tar.gz',
    # 'https://compneuro.net/datasets/md5sums.txt',
    # 'https://compneuro.net/datasets/shd_test.h5.gz',
    'https://compneuro.net/datasets/shd_test.h5.zip',
    # 'https://compneuro.net/datasets/shd_train.h5.gz',
    'https://compneuro.net/datasets/shd_train.h5.zip',
    # 'https://compneuro.net/datasets/ssc_test.h5.gz',
    # 'https://compneuro.net/datasets/ssc_test.h5.zip',
    # 'https://compneuro.net/datasets/ssc_train.h5.gz',
    # 'https://compneuro.net/datasets/ssc_train.h5.zip',
    # 'https://compneuro.net/datasets/ssc_valid.h5.gz',
    # 'https://compneuro.net/datasets/ssc_valid.h5.zip',
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
    print(labels)
    print(f['extra'].keys())
    print(f['extra']['meta_info'].keys())
    print(f['extra']['speaker'])
    print(f['extra']['keys'][2])

    # A quick raster plot for one of the samples
    if plot_it:
        n_subplots = 2
        fig, axs = plt.subplots(2, 1, gridspec_kw={'wspace': .0, 'hspace': 1.2}, figsize=(4, 4))

        # fig = plt.figure(figsize=(6, 6), gridspec_kw={'wspace': .0, 'hspace': 0.})
        idx = np.random.randint(len(times), size=n_subplots)

        bins = np.linspace(0, 1, 250)
        for k, ax in zip(idx, axs):
            # ax = plt.subplot(n_subplots, 1, i + 1)

            dense = np.zeros((700, 250))
            tr = times[k]
            which = tr < 1
            tr = tr[which]
            u = 699 - units[k][which]
            binned = np.digitize(tr, bins, right=True)  # times[k]
            dense[u, binned] = 1
            ax.pcolormesh(dense, cmap='Greys')
            # ax.scatter(binned, 700 - u, color="k", alpha=0.33, s=2)
            # ax.set_title("Label %i" % labels[k])
            ax.set_title(str(f['extra']['keys'][labels[k]].decode('utf-8')))

            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)
        axs[1].set_xlabel('time (ms)')
        axs[1].set_ylabel('channel')

        plt.savefig('heidelberg_spikes.png', bbox_inches='tight', dpi=500, transparent=True)
        plt.show()
        f.close()


class SpokenHeidelbergDigits(BaseGenerator):

    def __init__(
            self,
            data_dir,
            epochs=1,
            tvt='train',
            batch_size=32,
            repetitions=3,
            steps_per_epoch=None
    ):

        self.HEIDELBERGDIR = os.path.join(data_dir, 'SpikingHeidelbergDigits')
        shd_train_filename = os.path.join(self.HEIDELBERGDIR, "shd_train.h5")
        npy_shd_train_filename = os.path.join(self.HEIDELBERGDIR, "trainX_4ms.npy")
        if not os.path.exists(shd_train_filename):
            os.makedirs(self.HEIDELBERGDIR, exist_ok=True)
            download_and_unzip(data_links, self.HEIDELBERGDIR)

        if not os.path.exists(npy_shd_train_filename):
            for set in ['train', 'test']:
                original_filename = os.path.join(self.HEIDELBERGDIR, "shd_{}.h5".format(set))

                test_X, test_y = generate_dataset(original_filename, dt=4e-3)
                np.save(os.path.join(self.HEIDELBERGDIR, "{}X_4ms.npy".format(set)), test_X)
                np.save(os.path.join(self.HEIDELBERGDIR, "{}y_4ms.npy".format(set)), test_y)

        self.__dict__.update(batch_size=batch_size, tvt=tvt,
                             steps_per_epoch=steps_per_epoch, repetitions=repetitions)

        self.on_epoch_end()
        self.length = self.X.shape[1]
        self.in_dim = self.X.shape[2]
        self.out_dim = 20
        self.in_len = self.length * repetitions
        self.out_len = self.length * repetitions
        self.epochs = 450 if epochs == None else epochs

        self.batch_size = batch_size
        self.batch_index = 0

        self.steps_per_epoch = int(len(self.set) / self.batch_size) \
            if steps_per_epoch == None else steps_per_epoch

    def on_epoch_end(self):
        self.batch_index = 0
        path_X = lambda x: os.path.join(self.HEIDELBERGDIR, "{}X_4ms.npy".format(x))
        path_y = lambda x: os.path.join(self.HEIDELBERGDIR, "{}y_4ms.npy".format(x))

        if self.tvt == 'test':
            set_name = 'test'
            self.set = range(0, 2088)

        elif self.tvt in ['validation', 'val']:
            set_name = 'train'
            self.set = range(int(95 * 8155 / 100), 8155)

        elif self.tvt == 'train':
            set_name = 'train'
            self.set = range(0, int(95 * 8155 / 100))

        else:
            raise NotImplementedError

        self.X = np.load(path_X(set_name), mmap_mode='r')
        self.y = np.load(path_y(set_name), mmap_mode='r')
        self.random_indices = np.array(list(self.set))
        np.random.shuffle(self.random_indices)

    def data_generation(self):
        indices = self.random_indices[:self.batch_size]
        self.random_indices = self.random_indices[self.batch_size:]
        batch = self.X[indices]
        target = self.y[indices]
        target = np.repeat(target[..., None], self.length, 1)
        return {'input_spikes': batch, 'target_output': target, 'mask': 1.}


def test_generator():
    gen = SpokenHeidelbergDigits(
        data_dir=DATADIR,
        epochs=1,
        tvt='train',
        batch_size=32,
        repetitions=2,
        steps_per_epoch=None
    )
    print(gen.steps_per_epoch)
    for i in range(gen.steps_per_epoch):
        batch = gen.__getitem__()
        print(i, [b.shape for b in batch[0]])


if __name__ == '__main__':
    test_generator()
