

import sys, random, nltk

import numpy as np
import tensorflow as tf

from nltk.parse.generate import generate
from nltk.grammar import Nonterminal, CFG

from tensorflow.keras.preprocessing.sequence import pad_sequences

# grammar cannot have recursion!
from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator
from GenericTools.KerasTools.esoteric_tasks.nlp import Vocabulary, postprocessSentence
from GenericTools.KerasTools.esoteric_tasks.random_context_free import RandomGrammar
from GenericTools.KerasTools.esoteric_tasks.utils import unpackbits

np.set_printoptions(threshold=sys.maxsize)

grammar_string_simplest = """
        S -> 'hi' | 'this' 'is' 'a' 'long' 'sentence' | 'another' 'one'
        """

grammar_string_dogcat = """
        S -> NP VP | NP V
        VP -> V NP
        NP -> Det N
        Det -> 'a' | 'the'
        N -> 'dog' | 'cat'
        V -> 'chased' | 'saw'
        """

# grammar from Two Ways to Build a Thought: Distinct Forms of Compositional Semantic Representation across Brain Regions
nouns = ['moose', 'hawk', 'cow', 'goose', 'crow', 'hog']
nouns_string = "'" + "' | '".join(nouns) + "'"
verbs = ['passed', 'approached', 'attacked', 'bumped', 'surprised', 'frightened', 'noticed', 'detected']
verbs_string = "'" + "' | '".join(verbs) + "'"
grammar_string = """
        S -> NP V NP | NP V | NP PV 'by' NP | NP PV
        PV -> 'was' V
        NP -> Det N
        Det -> 'a' | 'the'
        N -> {}
        V -> {}
        """.format(nouns_string, verbs_string)


class NltkGrammarSampler(object):

    def __init__(self, grammar, outputTokens=False):

        if not isinstance(grammar, nltk.CFG):
            print('Using grammar file: %s' % (grammar))
            grammar = nltk.data.load('file:' + grammar)

        self.__dict__.update(grammar=grammar, outputTokens=outputTokens)

    def generate(self, n=0, start=None, depth=None):
        if not start:
            start = self.grammar.start()
        if depth is None:
            depth = sys.maxsize

        while True:
            tokens = self._generate_random([start], depth)
            if self.outputTokens:
                sentence = tokens
            else:
                sentence = ' '.join(tokens)

            yield sentence

    def _generate_random(self, items, depth):

        tokens = []
        try:
            for item in items:
                if isinstance(item, Nonterminal):
                    if depth > 0:
                        prods = self.grammar.productions(lhs=item)
                        if len(prods) > 0:
                            prod = random.choice(prods)
                            tokens.extend(self._generate_random(
                                prod.rhs(), depth - 1))
                else:
                    tokens.append(item)

        except RuntimeError as _error:
            if _error.message == "maximum recursion depth exceeded":
                raise RuntimeError("The grammar has rule(s) that yield infinite recursion!!")
            else:
                raise

        return tokens


def retreive_recall_time(output_class):
    bs = output_class.shape[0]

    # detect the index corresponding to <START>
    classes = output_class[:, :, 0]
    nz = np.nonzero(classes)
    time_period = int(len(nz[0]) / bs)
    seen = set()
    uniq = [True if x not in seen and not seen.add(x) else False for x in nz[0]]

    # the beginning of <RECALL> is preceeded by <RECALL> itself and <PAD> (2*time_period)
    recall_time = nz[1][uniq] - 2 * time_period - 1
    return recall_time


def _basicGenerator(grammar, batch_size=3):
    # sentences = []
    while True:
        yield [[' '.join(sentence)] for sentence in generate(grammar, n=batch_size)]


def indicesToOneHot(indices, num_tokens):
    return np.eye(num_tokens)[indices]


def getNoisyInput(keep, input_indices):
    if keep != 1.:
        assert keep <= 1.
        assert keep >= 0.
        keep_matrix = np.random.choice([0, 1], input_indices.shape, p=[1 - keep, keep])
        input_indices = input_indices * keep_matrix
    return input_indices


class CFG_NextTimestep_Generator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            grammar_string=grammar_string,
            epochs=1,
            n_in=5,
            number_states=4, number_words=2, n_loops=0, n_ors=2,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=1,
            keep=1.):
        'Initialization'
        self.__dict__.update(n_in=n_in,
                             steps_per_epoch=steps_per_epoch,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             keep=keep)

        if grammar_string == None:
            grammar_string = RandomGrammar(number_states=number_states, number_words=number_words, n_loops=n_loops,
                                           n_ors=n_ors)
        # print(grammar_string)
        self.grammar = CFG.fromstring(grammar_string)
        self.vocabulary = Vocabulary.fromGrammar(self.grammar)
        self.vocab_size = self.vocabulary.getMaxVocabularySize()
        self.id_to_word = {v: k for k, v in self.vocabulary.indicesByTokens.items()}

        self.PAD = self.vocabulary.padIndex
        self.START = self.vocabulary.startIndex
        self.END = self.vocabulary.endIndex
        self.MASK = self.vocabulary.unkIndex

        self.epochs = 100 if epochs == None else epochs
        self.steps_per_epoch = 10 if epochs == None else epochs


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.steps_per_epoch == None:
            return 10000
        return self.steps_per_epoch

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        list_indices = []
        for sentence in generate(self.grammar, n=self.batch_size):
            sentence = ' '.join(sentence)
            sentence = postprocessSentence(sentence)
            indices = [self.PAD, self.START] + \
                      self.vocabulary.sentenceToIndices(sentence) + \
                      [self.END]
            indices = indices[:self.maxlen]

            list_indices.append(indices)

        padded = pad_sequences(list_indices, maxlen=int(self.maxlen / 2) + 1)
        batch_size, time_length = padded.shape[0], padded.shape[1]
        add_neutral_phases = np.zeros((batch_size, time_length * 2), dtype=np.int32)
        add_neutral_phases[:, ::2] = padded

        # sentence to binary code
        binary_indices = unpackbits(add_neutral_phases, self.n_in)
        return binary_indices[:, :-2], binary_indices[:, 1:-1]


class CFG_AutoEncoding_Generator(CFG_NextTimestep_Generator):
    def __init__(
            self,
            random_period=7,
            n_hot=5,
            time_period=20,
            remove_oh_pad=True,
            **kwargs):
        super().__init__(**kwargs)

        self.grammar_generator = NltkGrammarSampler(self.grammar)

        self.random_period = random_period
        self.n_hot = n_hot
        self.time_period_in = time_period
        self.time_period_out = time_period
        self.time_period = time_period

        self.remove_oh_pad = remove_oh_pad

        self.in_dim = self.vocab_size if not self.remove_oh_pad else (self.vocab_size - 1)
        self.out_dim = self.vocab_size if not self.remove_oh_pad else self.vocab_size - 1
        self.in_len = 30 #self.maxlen
        self.out_len = 30 #self.maxlen

        self.max_rate_hz = 200
        self.min_rate_hz = 2
        self.RECALL = self.vocabulary.recIndex

    def sentenceToIndices(self, sentence):
        sentence = postprocessSentence(sentence)
        indices = [self.PAD, self.START] + \
                  self.vocabulary.sentenceToIndices(sentence) + \
                  [self.END, self.PAD]

        input_indices, output_indices = indices, indices
        return input_indices, output_indices

    def data_generation(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        list_in_idx = []
        list_out_idx = []
        sentences = []
        for sentence in self.grammar_generator.generate():
            sentences.append(sentence)
            input_indices, output_indices = self.sentenceToIndices(sentence)
            pause = np.random.choice(self.random_period)

            input_indices = (input_indices
                             + [self.PAD] * pause
                             + [self.RECALL]
                             + [self.PAD] * len(output_indices))
            output_indices = ([self.PAD] * len(input_indices)
                              + [self.PAD] * pause
                              + [self.PAD]
                              + output_indices)
            input_indices = np.repeat(input_indices, self.time_period_in, axis=0).tolist()
            output_indices = np.repeat(output_indices, self.time_period_out, axis=0).tolist()

            list_in_idx.append(input_indices)
            list_out_idx.append(output_indices)

            if len(list_in_idx) >= self.batch_size:
                break

        padded_in = pad_sequences(list_in_idx, maxlen=self.in_len, value=self.PAD)
        padded_out = pad_sequences(list_out_idx, maxlen=self.out_len, value=self.PAD)
        mask = padded_out != self.PAD

        oh_in = tf.keras.utils.to_categorical(padded_in, num_classes=self.vocab_size)
        oh_out = tf.keras.utils.to_categorical(padded_out, num_classes=self.vocab_size)

        if self.remove_oh_pad:
            oh_in = oh_in[:, :, 1:]
            oh_out = oh_out[:, :, 1:]

        # n-hot encoding
        # oh_in = np.repeat(oh_in, self.n_hot, axis=2)

        # input to binomial
        # probs = (oh_in * (self.max_rate_hz - self.min_rate_hz) + self.min_rate_hz) * 1e-3  # ms-1
        # in_spikes = np.random.binomial(1, probs).astype(np.float32)

        mask = np.repeat(1 * mask[..., None], self.vocab_size - 1, axis=-1)

        return {'input_spikes': oh_in[:, :self.maxlen, :], 'target_output': oh_out[:, :self.maxlen, :],
                'mask': mask[:, :self.maxlen], 'sentences': sentences
                }


def sentences_to_characteristics(sentences):
    q_list = []
    for s in sentences:
        qualities = []
        for noun in nouns:
            for verb in verbs:
                # print(s)
                # moose neuron
                moose = noun in s
                # detection neuron
                detection = verb in s
                # neuron moose as subject
                moose_as_subject = '{} '.format(noun) in s
                # moose as object
                moose_as_object = noun in s and not moose_as_subject
                # moose as detector
                moose_as_detector_active = '{} {}'.format(noun, verb) in s
                moose_as_detector_in_passive = verb in s and noun in s and not moose_as_detector_active
                moose_as_detector = moose_as_detector_active or moose_as_detector_in_passive
                # print(moose, detection, moose_as_subject, moose_as_object, moose_as_detector)
                qualities.extend([moose, detection, moose_as_subject, moose_as_object, moose_as_detector])
        q_list.append([q * 1 for q in qualities])

    q = np.array(q_list).astype(float)
    return q


class MergeSearch(CFG_AutoEncoding_Generator):
    def __init__(self, epochs=1, curriculum=True, return_characteristics=False, **kwargs):
        super().__init__(**kwargs)
        # the following random number is used only by MergeSearch()
        self.epochs = 15 if epochs == None else epochs
        self.curriculum = curriculum
        self.return_characteristics=return_characteristics
        self.epoch = 0
        self.batch_rand_n = np.random.rand()
        self.batch_types = ['mix', 'mask', 'recall', 'mix_adjacent']
        self.in_len = self.maxlen
        self.out_len = int(self.maxlen / self.time_period)

        self.time_period_in = self.time_period
        self.time_period_out = 1

        self.step = 0
        if self.curriculum:
            final_ps = np.array([.4, .4, .2])
            initial_ps = np.array([0, 0, 1])
            self.epochs_change = int(1 * epochs / 3)
            self.step = (final_ps - initial_ps) / (epochs - self.epochs_change)
            self.ps = initial_ps
        else:
            self.ps = np.array([.4, .4, .2])

        self.ps = self.ps / np.sum(self.ps)
        self.cs = np.cumsum(self.ps)
        self.batch_flag = self.batch_type()

    def batch_type(self):
        choice = np.sum(self.cs < self.batch_rand_n)
        batch_flag = self.batch_types[choice]
        return batch_flag

    def on_epoch_end(self):
        self.epoch += 1
        if self.curriculum:
            if self.epoch > self.epochs_change - 1:
                self.ps += self.step

    def sentenceToIndices(self, sentence):
        sentence = postprocessSentence(sentence)
        indices = self.vocabulary.sentenceToIndices(sentence)

        if self.batch_flag == self.batch_types[0]:
            # mix
            input_indices = np.random.permutation(indices).tolist()
            output_indices = indices

        elif self.batch_flag == self.batch_types[1]:
            # mask
            max_masking = .8
            x = np.array(indices.copy())

            rand_masked = np.random.rand()
            n_masked = int(len(x) * max_masking * rand_masked)
            idx = np.random.choice(range(len(x)), size=n_masked, replace=False)

            x[idx] = self.MASK

            input_indices = x.tolist()
            output_indices = indices

        elif self.batch_flag == self.batch_types[2]:
            # recall
            input_indices, output_indices = indices, indices

        elif self.batch_flag == self.batch_types[3]:
            # mix only two adjacent elements, or two leaves in the grammatical tree
            dets_idx = self.vocabulary.sentenceToIndices('the a')
            det_choices = [i for i, x in enumerate(indices) if x in dets_idx]
            i = np.random.choice(det_choices)

            new_indices = indices.copy()
            new_indices[i] = indices[i + 1]
            new_indices[i + 1] = indices[i]

            input_indices, output_indices = new_indices, indices
        else:
            raise NotImplementedError

        input_indices = [self.PAD, self.START] + input_indices + [self.END, self.PAD]
        output_indices = [self.PAD, self.START] + output_indices + [self.END, self.PAD]

        return input_indices, output_indices

    def data_generation(self):
        batch = super().data_generation()
        self.batch_rand_n = np.random.rand()

        # 2/5 the times mix the input, 2/5 the time mask the input, 1/5 recall
        self.batch_flag = self.batch_type()
        return batch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        if not self.return_characteristics:
            return (batch['input_spikes'], batch['mask']), (batch['target_output'], *[np.array(0)] * self.n_regularizations)
        else:
            q = sentences_to_characteristics(batch['sentences'])
            q = np.expand_dims(q, axis=1)
            q = np.repeat(q, self.maxlen, axis=1)
            return (batch['input_spikes'], batch['mask'], q), (np.zeros_like(q[:,0,0]),)




def test_merge():
    batch_size = 1
    generator = MergeSearch(grammar_string=grammar_string,
                            batch_size=batch_size,
                            steps_per_epoch=1,
                            n_hot=1,
                            time_period=1,
                            maxlen=20)
    # check REBER generator

    for _ in range(3):
        generator.batch_flag = 'mix_adjacent'
        batch = generator.data_generation()

        for k in batch.keys():
            print()
            print(batch[k])

def test_ae():

    batch_size = 1
    generator = CFG_AutoEncoding_Generator(grammar_string=grammar_string_dogcat,
                            batch_size=batch_size,
                            steps_per_epoch=1,
                            time_period=1,
                            maxlen=20)
    # check REBER generator
    vocabulary = generator.id_to_word
    print(vocabulary)
    for _ in range(3):
        generator.batch_flag = 'mix_adjacent'
        batch = generator.data_generation()
        print(batch.keys())
        print(batch['mask'])

if __name__ == '__main__':
    test_ae()
