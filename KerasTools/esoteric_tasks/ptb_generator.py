import os
import numpy as np
import tensorflow as tf

from GenericTools.StayOrganizedTools.download_utils import download_and_unzip
from GenericTools.KerasTools.esoteric_tasks import ptb_reader
from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'ptb'))
if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)

data_links = ['https://data.deepai.org/ptbdataset.zip']


class PTBInput(object):
    """The input data."""

    def __init__(self, batch_size, num_steps, data, name=None):
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)
        self.input_generator = ptb_reader.ptb_producer_np_eprop(data, batch_size, num_steps, name=name)


class PTBGenerator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            neutral_phase_length=0,
            repetitions=1,
            train_val_test='train',
            data_path=DATAPATH,
            char_or_word='char',
            pad_value=0.0,
            category_coding='onehot'):

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            neutral_phases=neutral_phase_length,
            repetitions=repetitions,
            train_val_test=train_val_test,
            data_path=data_path,
            char_or_word=char_or_word,
            pad_value=pad_value,
            category_coding=category_coding)

        self.in_len = maxlen * (repetitions + neutral_phase_length)
        self.out_len = maxlen

        self.on_epoch_end()

        self.in_dim = 1
        self.upp_pow = int(np.log(self.vocab_size + 1) / np.log(2)) + 1
        self.out_dim = self.vocab_size if self.category_coding == 'onehot' else self.upp_pow

        self.neutral_phase_length = neutral_phase_length
        self.category_coding = category_coding

        neutral_phases = np.zeros((batch_size, self.out_len))
        nr = np.repeat(neutral_phases, neutral_phase_length, axis=1)
        self.n_slices = np.split(nr, self.out_len, axis=1)

        self.epochs = 100 if epochs == None else epochs

    def on_epoch_end(self):
        raw_data = ptb_reader.ptb_raw_data(self.data_path, self.char_or_word)
        train_data, valid_data, test_data, self.vocab_size, word_to_id = raw_data
        self.id_to_word = {v: k for k, v in word_to_id.items()}

        if self.train_val_test == 'train':
            self.data = PTBInput(batch_size=self.batch_size, num_steps=self.out_len, data=train_data,
                                 name="TrainInput")
        elif self.train_val_test == 'val':
            self.data = PTBInput(batch_size=self.batch_size, num_steps=self.out_len, data=valid_data,
                                 name="ValidInput")
        elif self.train_val_test == 'test':
            self.data = PTBInput(batch_size=self.batch_size, num_steps=self.out_len, data=test_data,
                                 name="TestInput")
        else:
            raise NotImplementedError

        if self.steps_per_epoch == None:
            self.steps_per_epoch = self.data.epoch_size

    def data_generation(self):
        input_batch, output_batch, _ = next(self.data.input_generator)
        input_batch = np.repeat(input_batch, self.repetitions, axis=1)
        next_target_data = tf.keras.utils.to_categorical(output_batch, num_classes=self.vocab_size)

        # remove silent phase symbol
        next_in_data = input_batch[..., None]
        new_mask = np.ones((self.batch_size, self.out_len, self.out_dim))
        return {'input_spikes': next_in_data, 'target_output': next_target_data,
                'mask': new_mask}


def test_1():
    batch_size = 3
    generator = PTBGenerator(
        batch_size=batch_size,
        steps_per_epoch=1,
        neutral_phase_length=0,
        maxlen=10, )

    batch = generator.data_generation()
    # print(batch)
    for k in batch.keys():
        print(batch[k].shape)

    for k in batch.keys():
        print()
        print(batch[k])

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, figsize=(6, 10), sharex='all',
                             gridspec_kw={'hspace': 0})
    [ax.clear() for ax in axes]

    axes[0].pcolormesh(batch['input_spikes'][0].T, cmap='Greys')
    axes[1].pcolormesh(batch['target_output'][0].T, cmap='Greys')
    plt.show()


def test_2():
    batch_size = 3
    generator = PTBGenerator(
        batch_size=batch_size,
        steps_per_epoch=1,
        neutral_phase_length=0,
        char_or_word='word',
        maxlen=8, )

    batch = generator.data_generation()
    print(batch['input_spikes'].shape)
    print(batch['target_output'].shape)
    print(batch['target_output'].shape)
    print(batch['input_spikes'])
    print(batch['target_output'])


def test_check_2_contiguous_batch():
    batch_size = 10
    generator = PTBGenerator(
        batch_size=batch_size,
        steps_per_epoch=1,
        neutral_phase_length=0,
        char_or_word='char',
        maxlen=30, )

    i2t = generator.id_to_word
    print(i2t)

    batch_1 = generator.data_generation()
    batch_2 = generator.data_generation()
    print([''.join([i2t[i] for i in line]) for line in batch_1['input_spikes'][:, :, 0]])
    print([''.join([i2t[i] for i in line]) for line in batch_2['input_spikes'][:, :, 0]])
    print([''.join([i2t[i] for i in line]) for line in np.argmax(batch_1['target_output'], axis=2)])
    print([''.join([i2t[i] for i in line]) for line in np.argmax(batch_2['target_output'], axis=2)])


if __name__ == '__main__':
    # test_2()
    test_check_2_contiguous_batch()

    if len(os.listdir(DATAPATH)) == 0:
        download_and_unzip(data_links, DATAPATH)
