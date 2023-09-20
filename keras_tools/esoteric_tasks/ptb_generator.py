import os, json
import numpy as np
import tensorflow as tf

from pyaromatics.stay_organized.download_utils import download_and_unzip
from pyaromatics.keras_tools.esoteric_tasks import ptb_reader
from pyaromatics.keras_tools.esoteric_tasks.base_generator import BaseGenerator
from pyaromatics.keras_tools.esoteric_tasks.tensorlayer_ptb import ptb_iterator

# CDIR = os.path.dirname(os.path.realpath(__file__))
# DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'ptb'))
# os.makedirs(DATAPATH, exist_ok=True)

data_links = ['https://data.deepai.org/ptbdataset.zip']


class PTBGenerator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            neutral_phase_length=0,
            repetitions=1,
            train_val_test='train',
            char_or_word='char',
            pad_value=0.0,
            category_coding='onehot',
            config=''):

        data_dir = os.path.join(data_dir, 'ptb')
        os.makedirs(data_dir, exist_ok=True)

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            neutral_phases=neutral_phase_length,
            repetitions=repetitions,
            train_val_test=train_val_test,
            char_or_word=char_or_word,
            pad_value=pad_value,
            category_coding=category_coding,
            data_dir=data_dir,
            config=config)

        if len(os.listdir(data_dir)) == 0:
            download_and_unzip(data_links, data_dir)

        self.maxlen = maxlen
        self.in_len = maxlen * repetitions
        self.out_len = maxlen * repetitions

        if char_or_word == 'char':
            self.epochs = 50 if epochs == None else epochs
        elif char_or_word == 'word':
            self.epochs = 250 if epochs == None else epochs
        else:
            raise NotImplementedError

        self.on_epoch_end()

        self.in_dim = 1
        self.out_dim = self.vocab_size

        self.postprocess = lambda x: tf.keras.utils.to_categorical(x, num_classes=self.vocab_size) \
            if category_coding == 'onehot' else x

    def on_epoch_end(self):
        self.generator = generator_2 if not 'ptb1' in self.config else generator_1

        self.input_generator, self.id_to_word, len_data, self.vocab_size = self.generator(
            self.data_dir, self.char_or_word, self.train_val_test, self.batch_size, self.maxlen
        )

        if self.steps_per_epoch == None:
            self.steps_per_epoch = len_data // self.batch_size // self.maxlen - 1
            if not 'ptb1' in self.config:
                self.steps_per_epoch -= 1

        super().on_epoch_end()

    def data_generation(self):
        input_batch, output_batch, _ = next(self.input_generator)
        # next_target_data = tf.keras.utils.to_categorical(output_batch, num_classes=self.vocab_size)
        output_batch = self.postprocess(output_batch)

        # remove silent phase symbol
        next_in_data = input_batch[..., None]
        new_mask = np.ones((self.batch_size, self.out_len, self.out_dim))
        print(next_in_data.shape, output_batch.shape, new_mask.shape)
        return {'input_spikes': next_in_data, 'target_output': output_batch,
                'mask': new_mask}


def generator_1(data_path, char_or_word, train_val_test, batch_size, maxlen):
    raw_data = ptb_reader.ptb_raw_data(data_path, char_or_word)
    train_data, valid_data, test_data, _, word_to_id = raw_data
    id_to_word = {v: k for k, v in word_to_id.items()}

    if train_val_test == 'train':
        data = train_data
    elif train_val_test == 'val':
        data = valid_data
    elif train_val_test == 'test':
        data = test_data
    else:
        raise NotImplementedError

    input_generator = ptb_reader.ptb_producer_np_eprop(data, batch_size, maxlen)

    return input_generator, id_to_word, len(data), len(word_to_id)


def generator_2(data_path, char_or_word, train_val_test, batch_size, maxlen):
    # get vocabulary
    # DATAPATH = os.path.join(data_path, 'ptb', )

    vocab_path = os.path.join(data_path, 'vocab.json')
    if not os.path.exists(vocab_path):
        path = os.path.join(data_path, 'ptb.train.txt')

        with open(path) as f:
            contents = f.read()

        list_words = [w for w in contents.split(' ') if not w is '']
        vocabulary = np.unique(list_words)
        word_to_id = {v: i for i, v in enumerate(vocabulary)}
        id_to_word = {i: v for i, v in enumerate(vocabulary)}

        json.dump(word_to_id, open(vocab_path, "w"))
    else:
        with open(vocab_path) as f:
            word_to_id = json.load(f)
        vocabulary = list(word_to_id.keys())
        id_to_word = {v: k for k, v in word_to_id.items()}

    # get text

    tokenized_path = os.path.join(data_path, 'tokenized_{}.npy'.format(train_val_test))
    if not os.path.exists(tokenized_path):
        path = os.path.join(data_path, 'ptb.{}.txt'.format(train_val_test.replace('val', 'valid')))

        with open(path) as f:
            contents = f.read()

        list_words = [w for w in contents.split(' ') if not w is '']
        ptb_numbers = [word_to_id[w] for w in list_words]
        np.save(tokenized_path, ptb_numbers)
    else:
        ptb_numbers = np.load(tokenized_path)

    input_generator = ptb_iterator(ptb_numbers, batch_size, maxlen)
    return input_generator, id_to_word, len(ptb_numbers), len(vocabulary)


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

    fig, axes = plt.subplots(2, figsize=(6, 10), sharex='all', gridspec_kw={'hspace': 0})
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
    print([' '.join([i2t[i] for i in line]) for line in batch_1['input_spikes'][:, :, 0]])
    print([' '.join([i2t[i] for i in line]) for line in batch_2['input_spikes'][:, :, 0]])
    print([' '.join([i2t[i] for i in line]) for line in np.argmax(batch_1['target_output'], axis=2)])
    print([' '.join([i2t[i] for i in line]) for line in np.argmax(batch_2['target_output'], axis=2)])

    print(batch_1['input_spikes'][:, :, 0].shape)
    batch = batch_1['input_spikes'][:, :, 0]
    print(batch)

    oh = tf.one_hot(batch, depth=generator.vocab_size)[0].numpy().T[:50, :]
    print(oh.shape)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, figsize=(4, 3), )
    c = 'Greys'
    axs.pcolormesh(oh, cmap=c)
    axs.set_xlabel('time')
    axs.set_ylabel('word index')
    for pos in ['right', 'left', 'bottom', 'top']:
        axs.spines[pos].set_visible(False)

    axs.set_ylim([20, 50])
    plt.savefig('ptb_spikes.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # if len(os.listdir(DATAPATH)) == 0:
    #     download_and_unzip(data_links, DATAPATH)
    #
    # test_2()
    test_check_2_contiguous_batch()
