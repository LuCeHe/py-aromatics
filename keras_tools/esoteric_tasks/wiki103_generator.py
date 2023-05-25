from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import os, itertools, json
import numpy as np
import tensorflow as tf

from pyaromatics.stay_organized.download_utils import download_and_unzip

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wiki103'))

if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)

DATAPATH = os.path.join(DATAPATH, 'wikitext-103-raw')
sets = ['train', 'test', 'valid']  # ['train', 'test', 'valid']

max_text_in_txt = int(1e4)  # int(5e6)

def initialize_wiki103_dataset():
    data_links = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip']
    download_and_unzip(data_links, DATAPATH)

    # organize data to avoid loading all on RAM
    for s in sets:
        s_folder = os.path.join(DATAPATH, s)
        if not os.path.isdir(s_folder): os.mkdir(s_folder)
        length_document = 0
        words = []

        raw_file = os.path.join(DATAPATH, 'wiki.{}.raw'.format(s))

        # if destination folders are empty, fill them
        if len(os.listdir(s_folder)) == 0:

            with open(raw_file, "r", encoding="utf-8") as f:
                l = f.readlines()
                full_l = ''.join(l).replace('\n', ' ').replace(' ', '_').replace('', ' ')
                chunks = [full_l[i:i + max_text_in_txt] for i in range(0, len(full_l), max_text_in_txt)]

                length_document += len(full_l.split())
                # FIXME: set == 'train'
                if s == 'train':
                    words = np.unique(full_l.split()).tolist()
                    # np.save(os.path.join(DATAPATH, 'words.npy'), words)
                    json_dict = json.dumps(words)
                    with open(os.path.join(DATAPATH, 'words.json'), "w") as f:
                        f.write(json_dict)
                    del words, json_dict
                del full_l

                os.mkdir(os.path.join(s_folder, 'data'))
                for i, t in enumerate(chunks):
                    char_file = os.path.join(s_folder, 'data/t_{}.txt'.format(i))
                    with open(char_file, 'w', encoding="utf-8") as g:
                        g.write(t)

            if not os.path.isfile(char_file + '.properties.json'):
                properties = {'length_document': length_document}
                json_dict = json.dumps(properties)
                with open(os.path.join(DATAPATH, 'properties_{}.json'.format(s)), "w") as f:
                    f.write(json_dict)


def batch_iterator(batch, int_vectorize_layer, maxlen):
    nps = int_vectorize_layer(batch)
    n_steps = int(nps.shape[1] / maxlen)
    i = 0
    while i < n_steps:
        x = nps[:, i * maxlen:(i + 1) * maxlen]
        i += 1
        yield x


def idx2sentences(batch, int_vectorize_layer):
    sentences = []
    for sample in batch:
        sentence = ''.join([int_vectorize_layer.get_vocabulary()[i] for i in sample])
        sentences.append(sentence)
    return sentences


def test():
    int_vectorize_layer = TextVectorization(max_tokens=5000, output_mode='int', standardize=None,
                                            output_sequence_length=max_text_in_txt, )

    with open(os.path.join(DATAPATH, 'words.json')) as json_file:
        words = json.load(json_file)

    int_vectorize_layer.set_vocabulary(words)

    s = 'valid'
    SETDIR = os.path.join(DATAPATH, s)

    file_paths = [os.path.join(SETDIR + r'/data', d) for d in os.listdir(SETDIR + r'/data')]
    dataset = tf.data.TextLineDataset(file_paths).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  #

    batch_size = 64
    maxlen = 500

    n_files = len(file_paths)

    iterator = iter(dataset.batch(batch_size))
    n_iterators = int(n_files / batch_size) + 1
    n_steps = int(max_text_in_txt / maxlen)
    print(n_iterators, n_steps)
    for _ in range(n_iterators):
        batch = next(iterator)
        # print(len(iterator))
        generator = batch_iterator(batch, int_vectorize_layer, maxlen=maxlen)

        for j in range(n_steps):
            b = next(generator)
            print(j, b.shape)
            # print(b)
            # print(i, idx2sentences(b))


class Wiki103Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            neutral_phase_length=1,
            repetitions=2,
            train_val_test='valid',
            char_or_word='char',
            pad_value=0.0):

        tmp_maxlen = int(maxlen / (repetitions + neutral_phase_length))
        initialize_wiki103_dataset()

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            tmp_maxlen=tmp_maxlen,
            neutral_phases=neutral_phase_length,
            repetitions=repetitions,
            train_val_test=train_val_test,
            char_or_word=char_or_word,
            pad_value=pad_value,
            n_regularizations=0, )
        self.on_epoch_end()
        self.int_vectorize_layer = TextVectorization(max_tokens=5000, output_mode='int', standardize=None,
                                                     output_sequence_length=max_text_in_txt, )

        with open(os.path.join(DATAPATH, 'words.json')) as json_file:
            words = json.load(json_file)

        self.int_vectorize_layer.set_vocabulary(words)

        self.id_to_word = {v: k for v, k in enumerate(self.int_vectorize_layer.get_vocabulary())}
        vocab_size = len(self.int_vectorize_layer.get_vocabulary())
        self.in_dim = vocab_size
        self.out_dim = vocab_size
        self.in_len = int(max_text_in_txt / 20) * repetitions
        self.out_len = int(max_text_in_txt / 20)
        self.epochs = 1 if epochs == None else epochs


    def on_epoch_end(self):
        self.SETDIR = os.path.join(DATAPATH, self.train_val_test.replace('val', 'valid'))
        file_paths = [os.path.join(self.SETDIR + r'/data', d) for d in os.listdir(self.SETDIR + r'/data')]
        dataset = tf.data.TextLineDataset(file_paths).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  #

        n_files = len(file_paths)

        self.iterator = iter(dataset.batch(self.batch_size))
        self.n_out_txt = int(n_files / self.batch_size) + 1
        self.n_in_txt = int(max_text_in_txt / self.maxlen)
        self.in_txt_step = self.n_in_txt + 2
        self.out_txt_step = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.steps_per_epoch is None:
            steps_per_epoch = self.n_out_txt * self.n_in_txt
        else:
            steps_per_epoch = self.steps_per_epoch
        return steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        return (batch['input_spikes'], batch['mask']), (batch['target_output'], *[np.array(0)] * self.n_regularizations)

    def indeces2sentences(self, batch):
        return idx2sentences(batch, self.int_vectorize_layer)

    def data_generation(self):

        if self.in_txt_step >= self.n_in_txt:
            self.in_txt_step = 0
            self.batch = next(self.iterator)
            if self.out_txt_step >= self.n_out_txt:
                self.out_txt_step = 0

        # print(len(iterator))
        generator = batch_iterator(self.batch, self.int_vectorize_layer, maxlen=self.out_len + 1)
        b = next(generator)
        in_data = b[:, :-1]
        target_data = b[:, 1:]

        # repeat input
        in_data = np.repeat(in_data, self.repetitions, axis=1)

        # to categorical
        self.in_txt_step += 1
        self.out_txt_step += 1

        in_data = tf.keras.utils.to_categorical(in_data, num_classes=self.in_dim)
        target_data = tf.keras.utils.to_categorical(target_data, num_classes=self.out_dim)
        new_mask = np.ones((self.batch_size, self.out_len, self.out_dim))

        return {'input_spikes': in_data, 'target_output': target_data,
                'mask': new_mask}
