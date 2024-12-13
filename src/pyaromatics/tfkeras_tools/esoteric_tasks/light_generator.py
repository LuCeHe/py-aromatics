import os

import numpy as np
import pandas as pd
import tensorflow as tf

from transformers import BlenderbotSmallTokenizer

from innocent_explorations.chatbot.light_dataset import data_splits

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', '..', '..', '..', 'data', 'light_dialogue'))

mname = 'facebook/blenderbot-90M'


class LightGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=3,
            steps_per_epoch=1,
            encoder_maxlen=512,
            decoder_maxlen=512,
            data_split='valid',
            data_path=DATAPATH,
            shuffle=True,
    ):

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            encoder_maxlen=encoder_maxlen,
            decoder_maxlen=decoder_maxlen,
            data_split=data_split,
            shuffle=shuffle,
        )

        assert data_split in data_splits

        h5d = f'speech_{data_split}.h5' if not data_split == 'unseen' else 'speech_test_unseen.h5'
        self.h5path = os.path.join(data_path, h5d)

        if data_path is None:
            raise ValueError("Specify the data_path where you want the data to be saved!")

        self.on_epoch_end()

        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

        self.pad_idx = self.tokenizer.pad_token_id
        self.start_idx = self.tokenizer.bos_token_id
        self.end_idx = self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.vocab_size

        self.knowledge_keys = ['persona', 'dialogue_history', 'description']

        self.epochs = 50 if epochs < 0 else epochs
        self.steps_per_epoch = int(self.n_samples / self.batch_size) if steps_per_epoch < 0 else steps_per_epoch

    def on_epoch_end(self):

        if hasattr(self, 'data'):
            del self.data
        self.data = pd.read_hdf(self.h5path, 'df')  # load it
        self.n_samples = len(self.data['answer'])

        self.random_indices = np.random.choice(self.n_samples, self.n_samples, replace=False) \
            if self.shuffle else range(self.n_samples)

    def data_generation(self, index=None):

        if index is None:
            index = np.random.randint(0, self.steps_per_epoch)
        batch_indices = self.random_indices[index * self.batch_size:(index + 1) * self.batch_size]

        data = self.data.iloc[batch_indices]
        targets = data['answer']
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets, padding="post", value=self.pad_idx)

        input_targets = targets[:, :-1]
        output_targets = targets[:, 1:]

        knowledge = {k: tf.keras.preprocessing.sequence.pad_sequences(
            data[k], value=self.pad_idx
        )
                     for k in self.knowledge_keys
                     }

        maxlen = max([knowledge[k].shape[1] for k in self.knowledge_keys])

        knowledge = {
            k: np.concatenate([
                self.pad_idx * np.ones((self.batch_size, maxlen - v.shape[1])), v
            ], axis=1)[..., :self.encoder_maxlen]
            for k, v in knowledge.items()
        }


        return {
            **knowledge,
            'input_targets': input_targets[..., :self.decoder_maxlen],
            'output_targets': output_targets[..., :self.decoder_maxlen]
        }

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation(index)
        return [
                   batch['persona'],
                   batch['dialogue_history'],
                   batch['description'],
                   batch['input_targets'],
                   batch['output_targets']
               ],batch['output_targets']


if __name__ == '__main__':
    gen = LightGenerator(
        epochs=1,
        batch_size=4,
        steps_per_epoch=1,
        encoder_maxlen=512,
        decoder_maxlen=512,
        data_split='valid',
    )

    batch = gen.__getitem__()
    print([b.shape for b in batch[0]])
