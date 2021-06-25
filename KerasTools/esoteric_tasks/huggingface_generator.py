import tensorflow as tf
import os
import numpy as np

from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator
from transformers import GPT2Tokenizer
import datasets

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data'))

if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)


class HuggingfaceGenerator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=128,
            train_val_test='val',
            dataset_name='lambada',
            mode='language_modeling'):

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            train_val_test=train_val_test,
            n_regularizations=0, )

        assert mode in ['language_modeling', 'distillation']
        assert dataset_name in ['wikitext-2', 'lambada', 'bookcorpus']
        print(dataset_name)
        self.old = True if mode == 'distillation' else False
        dataset_path = os.path.join(DATAPATH, dataset_name)
        set_tokenized_path = os.path.join(dataset_path, 'tokenized_{}_{}'.format(train_val_test, maxlen))

        self.mode = mode
        self.out_len = maxlen  # 390 - 1
        self.in_len = maxlen

        self.dset = datasets.load_from_disk(set_tokenized_path)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.lines = len(self.dset)
        self.vocab_size = tokenizer.vocab_size
        self.id_to_word = [k for k, _ in tokenizer.get_vocab().items()]
        # self.encoded_dataset = dataset.map(self.string2tokens, batched=True)

        self.on_epoch_end()
        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(self.lines / self.batch_size) - 1

        self.in_dim = 1
        self.out_dim = self.vocab_size

        # self.epochs = 20 if epochs == None else epochs

    def on_epoch_end(self):
        self.batch_i = 0
        self.dataset = self.dset.shuffle()

    def data_generation(self):
        bi = self.batch_size * self.batch_i
        bf = self.batch_size * (self.batch_i + 1)

        batch = np.array(self.dset['input_ids'][bi:bf])

        input = batch[:, :-1]
        output = batch[:, 1:]
        output = tf.keras.utils.to_categorical(output, num_classes=self.vocab_size)

        self.batch_i += 1

        return {'input_spikes': input, 'target_output': output,
                'mask': np.ones_like(input)}


