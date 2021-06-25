import tensorflow as tf
import os

from datasets import load_dataset as true_load_dataset
# from transformers import MarianTokenizer
import numpy as np

from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator
from transformers import GPT2Tokenizer
import datasets

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wikitext2'))
TOKPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'gpt2_tokenizer'))

if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)
if not os.path.isdir(TOKPATH): os.mkdir(TOKPATH)


def download_dataset():
    # import datasets
    # data = datasets.load_dataset(...)
    # data.save_to_disk(/YOUR/DATASET/DIR)
    # copy the dir from online to the offline machine
    # (offline machine)
    # import datasets
    # data = datasets.load_from_disk(/SAVED/DATA/DIR)

    for s in ['train', 'validation', 'test']:  # , 'validation[:10]', 'validation[10:]']:
        d = os.path.abspath(os.path.join(DATAPATH, s))
        if not os.path.isdir(d):
            os.mkdir(d)
        dataset = true_load_dataset('wikitext', 'wikitext-2-v1', split=s, cache_dir=d)
        dataset = dataset.filter(lambda example: not example['text'] == '')

        dataset.save_to_disk(d)
        print('wikitext-2-v1', s, len(dataset))


class Wiki2Generator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            train_val_test='train'):
        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            train_val_test=train_val_test,
            n_regularizations=0, )

        self.out_len = None  # 390 - 1
        self.in_len = None

        # dataset = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='validation')
        self.dataset = datasets.load_from_disk(DATAPATH + '/' + train_val_test)
        self.tokenizer = GPT2Tokenizer.from_pretrained(TOKPATH)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.lines = len(self.dataset)
        self.vocab_size = self.tokenizer.vocab_size
        self.id_to_word = [k for k, _ in self.tokenizer.get_vocab().items()]
        # self.encoded_dataset = dataset.map(self.string2tokens, batched=True)

        self.on_epoch_end()
        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(self.lines / self.batch_size) - 1
        else:
            if 'val' in train_val_test:
                self.steps_per_epoch = 4

        self.in_dim = 1
        self.out_dim = self.vocab_size

        # self.epochs = 20 if epochs == None else epochs

    def on_epoch_end(self):
        self.batch_i = 0
        self.dataset = self.dataset.shuffle()

    def data_generation(self):
        bi = self.batch_size * self.batch_i
        bf = self.batch_size * (self.batch_i + 1)

        batch = self.dataset['text'][bi: bf]
        generated = self.tokenizer(batch, return_tensors="tf", padding=True, truncation=True)

        attention_mask = generated['attention_mask'][..., None][:, 1:]
        input = generated['input_ids'][:, :-1]
        output = generated['input_ids'][:, 1:]
        output = tf.keras.utils.to_categorical(output, num_classes=self.vocab_size)

        self.batch_i += 1

        return {'input_spikes': input, 'target_output': output,
                'mask': attention_mask}


def generator_wiki(batch_size, split='validation[:10%]', steps_per_epoch=None):
    # lines:
    # wikitext-2-v1 train 36718
    # wikitext-2-v1 validation 3760
    # wikitext-2-v1 test 4358

    tokenizer = GPT2Tokenizer.from_pretrained(TOKPATH)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split, cache_dir=DATAPATH)
    dataset = datasets.load_from_disk(DATAPATH + '/' + split)
    dataset = dataset.filter(lambda example: not example['text'] == '')

    lines = len(dataset)
    n_batches = lines // batch_size

    i = 0
    if steps_per_epoch is None: steps_per_epoch = np.inf

    while i < steps_per_epoch:
        batch = dataset['text'][i * batch_size: (i + 1) * batch_size]
        generated = tokenizer(batch, return_tensors="tf", padding=True, truncation=True)

        attention_mask = generated['attention_mask'][..., None][:, 1:]
        input = generated['input_ids'][:, :-1]
        output = generated['input_ids'][:, 1:]
        output = tf.keras.utils.to_categorical(output, num_classes=tokenizer.vocab_size)
        i += 1

        yield (input, attention_mask), output


if __name__ == '__main__':
    # download_models()
    # study_pratrained_layers()
    download_dataset()
