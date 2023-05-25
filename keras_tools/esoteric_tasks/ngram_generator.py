import numpy as np
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pyaromatics.LeanguageTreatmentTools.nlp import Vocabulary
from sg_design_lif.generate_data.utils import unpackbits

nltk.download('gutenberg')
nltk.download('punkt')

from nltk.corpus import gutenberg
from nltk.util import ngrams

from sg_design_lif.generate_data.kneser_ney import KneserNeyLM



class NgramGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            batch_size=2,
            steps_per_epoch=100,
            n_gram_length=6,
            n_in=32,
            max_len=30,
            fileid='bible-kjv.txt'):
        self.__dict__.update(batch_size=batch_size,
                             steps_per_epoch=steps_per_epoch,
                             n_in=n_in,
                             max_len=max_len
                             )
        gut_ngrams = (
            ngram for sent in gutenberg.sents(fileid) for ngram in ngrams(sent, n_gram_length,
                                                                          pad_left=True,
                                                                          pad_right=True,
                                                                          right_pad_symbol='</s>',
                                                                          left_pad_symbol='<s>'))
        self.lm = KneserNeyLM(n_gram_length, gut_ngrams)
        words = gutenberg.words(fileid)
        self.vocabulary = Vocabulary(words)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        X, y = self.data_generation()
        return X, y

    def on_epoch_end(self):
        pass

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        list_indices = []
        for _ in range(self.batch_size):
            sentence = self.lm.generate_sentence()

            # sentence to indices
            indices = self.vocabulary.sentenceToIndices(sentence)
            list_indices.append(indices)

        padded = pad_sequences(list_indices, maxlen=int(self.max_len/2)+1)
        batch_size, time_length = padded.shape[0], padded.shape[1]
        add_neutral_phases = np.zeros((batch_size, time_length * 2), dtype=np.int32)
        add_neutral_phases[:, ::2] = padded

        # sentence to binary code
        binary_indices = unpackbits(add_neutral_phases, self.n_in)
        return binary_indices[:, :-2], binary_indices[:, 1:-1]


if __name__ == '__main__':
    gen = NgramGenerator()
    batch = gen.__getitem__()
    print([b.shape for b in batch])
