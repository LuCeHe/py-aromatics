# Copyright (c) 2018, 
#
# authors: Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import gzip
import string
from subprocess import Popen, PIPE

import numpy as np
from nltk import CFG
from nltk.parse.generate import generate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# grammar cannot have recursion!
from GenericTools.LeanguageTreatmentTools.nlp import Vocabulary, NltkGrammarSampler, tokenize

grammar = CFG.fromstring("""
                         S -> NP VP | NP V
                         VP -> V NP
                         NP -> Det N
                         Det -> 'a' | 'the'
                         N -> 'dog' | 'cat'
                         V -> 'chased' | 'saw'
                         """)


def _basicGenerator(grammar, batch_size=3):
    # sentences = []
    while True:
        yield [[' '.join(sentence)] for sentence in generate(grammar, n=batch_size)]


def sentencesToCharacters(sentences):
    """
    The input:
        [['the dog  chased a cat'],
        ['the dog sat in a cat'],
        ['the cat sat on a cat']]
    The output:
        [['t', 'h', 'e', ' ', 'd', 'o', ...],
        ['t', 'h', ...],
        ['t', ...]]
    """

    assert isinstance(sentences, list)
    assert isinstance(sentences[0], list)
    assert isinstance(sentences[0][0], str)

    charSentences = [list(sentence[0]) for sentence in sentences]

    return charSentences


def _charactersGenerator(grammar, batch_size=5):
    # FIXME: the generator is not doing what it should be doing,
    # since ell the batches are the same
    # but it's fine for now since this is only a toy scenario

    while True:
        sentences = [[' '.join(sentence)] for sentence in generate(grammar, n=batch_size)]
        yield sentencesToCharacters(sentences)


def _charactersNumsGenerator(grammar, batch_size=5):
    tokens = sorted(list(string.printable))
    # print(tokens)

    vocabulary = Vocabulary(tokens)
    # print(vocabulary.getMaxVocabularySize())

    # FIXME: the generator is not doing what it should be doing, 
    # since ell the batches are the same
    # but it's fine for now since this is only a toy scenario

    while True:
        sentences = [[' '.join(sentence)] for sentence in generate(grammar, n=batch_size)]
        # print(sentences)
        sentencesCharacters = sentencesToCharacters(sentences)
        sentencesIndices = [vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentencesCharacters]
        padded_indices = pad_sequences(sentencesIndices)
        yield padded_indices


class c2n_generator(object):

    def __init__(self, grammar, batch_size=5, maxlen=None, categorical=False):

        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.categorical = categorical

        tempVocabulary = Vocabulary.fromGrammar(grammar)
        tokens = tempVocabulary.tokens[1:]
        tokens = set(sorted(' '.join(tokens)))
        self.vocabulary = Vocabulary(tokens)
        # self.tokens = sorted(list(string.printable))
        # self.vocabulary = Vocabulary(self.tokens)
        print('')
        print(self.vocabulary.indicesByTokens)
        print('')
        # +1 to take into account padding
        self.vocabSize = len(self.vocabulary.indicesByTokens)
        self.startId = self.vocabulary.tokenToIndex(self.vocabulary.endToken)

        # self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)

    def generator(self):
        while True:
            sentences = [[''.join(sentence)] for sentence in self.sampler.generate(self.batch_size)]
            sentencesCharacters = sentencesToCharacters(sentences)
            # offset=1 to take into account padding
            indices = [self.vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentencesCharacters]

            # before: good for sequence2sequence
            # indices = pad_sequences(indices, maxlen=self.maxlen-2, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            # indices = pad_sequences(indices, maxlen=self.maxlen-1, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            # in_indices  = pad_sequences(indices, maxlen=self.maxlen, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            # out_indices = pad_sequences(indices, maxlen=self.maxlen, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')

            # after: not good, just to pick one of the two
            # indices = pad_sequences(indices, maxlen=self.maxlen-2, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            # indices = pad_sequences(indices, maxlen=self.maxlen-1, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            in_indices = pad_sequences(indices, maxlen=self.maxlen,
                                       value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            out_indices = pad_sequences(indices, maxlen=self.maxlen,
                                        value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')

            if not self.categorical:
                yield in_indices, out_indices
            else:
                categorical_indices = to_categorical(out_indices, num_classes=self.vocabSize)
                yield in_indices, categorical_indices

    def indicesToSentences(self, indices, offset=0):
        if not isinstance(indices[0][0], int):
            indices = [[int(i) for i in list_idx] for list_idx in indices]
        return self.vocabulary.indicesToSentences(indices, offset=offset)


class next_character_generator(object):

    def __init__(self, grammar, batch_size=5, maxlen=None, categorical=False):

        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.categorical = categorical

        tempVocabulary = Vocabulary.fromGrammar(grammar)
        tokens = tempVocabulary.tokens[1:]
        tokens = set(sorted(' '.join(tokens)))
        self.vocabulary = Vocabulary(tokens)
        # self.tokens = sorted(list(string.printable))
        # self.vocabulary = Vocabulary(self.tokens)
        print('')
        print(self.vocabulary.indicesByTokens)
        print('')
        # +1 to take into account padding
        self.vocabSize = len(self.vocabulary.indicesByTokens)
        self.startId = self.vocabulary.tokenToIndex(self.vocabulary.endToken)

        # self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)

    def generator(self):
        while True:
            sentences = [[''.join(sentence)] for sentence in self.sampler.generate(self.batch_size)]
            sentencesCharacters = sentencesToCharacters(sentences)
            # offset=1 to take into account padding
            # indices = [self.vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentencesCharacters]

            import numpy as np

            list_input = []
            list_output = []
            for i, listOfTokens in enumerate(sentencesCharacters):
                indices = self.vocabulary.tokensToIndices(listOfTokens)
                sentence_len = len(indices)
                next_token_pos = np.random.randint(sentence_len)
                input_indices = indices[:next_token_pos]
                next_token = [indices[next_token_pos]]

                list_input.append(input_indices)
                list_output.append(next_token)

            in_indices = pad_sequences(list_input, maxlen=self.maxlen,
                                       value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            out_indices = np.array(list_output)

            if not self.categorical:
                yield in_indices, out_indices
            else:
                categorical_indices = to_categorical(out_indices, num_classes=self.vocabSize)
                yield in_indices, categorical_indices

    def indicesToSentences(self, indices, offset=0):
        if not isinstance(indices[0][0], int):
            indices = [[int(i) for i in list_idx] for list_idx in indices]
        return self.vocabulary.indicesToSentences(indices, offset=offset)


class next_word_generator(object):

    def __init__(self, grammar, batch_size=5, maxlen=None, categorical=False):

        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.categorical = categorical

        self.vocabulary = Vocabulary.fromGrammar(grammar)

        # self.tokens = sorted(list(string.printable))
        # self.vocabulary = Vocabulary(self.tokens)
        print('')
        print(self.vocabulary.indicesByTokens)
        print('')
        # +1 to take into account padding
        self.vocabSize = len(self.vocabulary.indicesByTokens)
        self.startId = self.vocabulary.tokenToIndex(self.vocabulary.endToken)

        # self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)

    def generator(self):
        while True:
            sentences = [sentence.split(' ') for sentence in self.sampler.generate(self.batch_size)]
            # sentencesCharacters = sentencesToCharacters(sentences)
            # offset=1 to take into account padding
            indices = [self.vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentences]

            list_input = []
            list_output = []
            for listOfIndices in indices:
                sentence_len = len(listOfIndices)
                next_token_pos = np.random.randint(sentence_len)
                input_indices = listOfIndices[:next_token_pos]
                next_token = [listOfIndices[next_token_pos]]

                list_input.append(input_indices)
                list_output.append(next_token)

            in_indices = pad_sequences(list_input, maxlen=self.maxlen,
                                       value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            out_indices = np.array(list_output)

            if not self.categorical:
                yield in_indices, out_indices
            else:
                categorical_indices = to_categorical(out_indices, num_classes=self.vocabSize)
                yield in_indices, categorical_indices

    def indicesToSentences(self, indices, offset=0):
        if not isinstance(indices[0][0], int):
            indices = [[int(i) for i in list_idx] for list_idx in indices]
        return self.vocabulary.indicesToSentences(indices, offset=offset)


class w2n_generator(object):

    def __init__(self, grammar, batch_size=5, maxlen=None, categorical=False):

        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.categorical = categorical

        self.vocabulary = Vocabulary.fromGrammar(grammar)

        print('')
        print(self.vocabulary.indicesByTokens)
        print('')
        # +1 to take into account padding
        self.vocabSize = len(self.vocabulary.indicesByTokens)
        self.startId = self.vocabulary.tokenToIndex(self.vocabulary.endToken)

        # self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)

    def generator(self):
        while True:
            sentences = [sentence.split(' ') for sentence in self.sampler.generate(self.batch_size)]
            # sentencesCharacters = sentencesToCharacters(sentences)
            # offset=1 to take into account padding
            indices = [self.vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentences]

            # before: good for sequence2sequence
            # indices = pad_sequences(indices, maxlen=self.maxlen-2, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            # indices = pad_sequences(indices, maxlen=self.maxlen-1, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            # in_indices  = pad_sequences(indices, maxlen=self.maxlen, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            # out_indices = pad_sequences(indices, maxlen=self.maxlen, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')

            # after: not good, just to pick one of the two
            # indices = pad_sequences(indices, maxlen=self.maxlen-2, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')
            # indices = pad_sequences(indices, maxlen=self.maxlen-1, value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            in_indices = pad_sequences(indices, maxlen=self.maxlen,
                                       value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='pre')
            out_indices = pad_sequences(indices, maxlen=self.maxlen,
                                        value=self.vocabulary.indicesByTokens[self.vocabulary.endToken], padding='post')

            if not self.categorical:
                yield in_indices, out_indices
            else:
                categorical_indices = to_categorical(out_indices, num_classes=self.vocabSize)
                yield in_indices, categorical_indices

    def indicesToSentences(self, indices, offset=0):
        if not isinstance(indices[0][0], int):
            indices = [[int(i) for i in list_idx] for list_idx in indices]
        return self.vocabulary.indicesToSentences(indices, offset=offset)


def generateFromGzip(gzipDatasetFilepath, batchSize):
    # read last sentence to reinitialize the generator once it's found
    this_gzip = Popen(['gzip', '-dc', gzipDatasetFilepath], stdout=PIPE)
    tail = Popen(['tail', '-1'], stdin=this_gzip.stdout, stdout=PIPE)
    last_sentence = tail.communicate()[0][:-2]

    f = gzip.open(gzipDatasetFilepath, 'rb')

    while True:
        sentences = []
        for line in f:
            sentence = line.strip().decode("utf-8")
            sentences.append(sentence)

            if sentence == last_sentence:
                sentences = []
                f = gzip.open(gzipDatasetFilepath, 'rb')

            if len(sentences) >= batchSize:
                batch = sentences
                sentences = []
                yield batch


def SentenceToIndicesGenerator(sentence_generator, vocabulary, maxSentenceLen=None):
    PAD = vocabulary.indicesByTokens[vocabulary.padToken]
    START = vocabulary.indicesByTokens[vocabulary.startToken]
    END = vocabulary.indicesByTokens[vocabulary.endToken]
    while True:
        sentences = next(sentence_generator)
        # NOTE: use offset to reserve a place for the masking symbol at
        # zero
        indices = [[PAD, START] + vocabulary.tokensToIndices(tokenize(sentence)) + [END]
                   for sentence in sentences]

        padded = pad_sequences([tokens[:maxSentenceLen] for tokens in indices],
                               # maxlen=maxSentenceLen,
                               value=PAD,
                               padding='pre')
        yield padded


def IndicesToNextStepGenerator(indices_generator, vocabSize=None):
    while True:
        indices = next(indices_generator)
        maxlen = indices.shape[1]
        column = np.random.randint(low=1, high=maxlen)
        model_input = indices[:, :column]
        model_output = indices[:, column, np.newaxis]
        if isinstance(vocabSize, int):
            model_output = to_categorical(model_output, num_classes=vocabSize)
        yield model_input, model_output


def GzipToNextStepGenerator(gzip_filepath, grammar_filepath, batchSize, maxSentenceLen=None):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    vocabSize = vocabulary.getMaxVocabularySize()

    generatorSentences = generateFromGzip(
        gzipDatasetFilepath=gzip_filepath,
        batchSize=batchSize
    )
    generatorIndices = SentenceToIndicesGenerator(
        sentence_generator=generatorSentences,
        vocabulary=vocabulary,
        maxSentenceLen=maxSentenceLen
    )
    generatorNextStep = IndicesToNextStepGenerator(
        generatorIndices,
        vocabSize
    )
    while True:
        generation = next(generatorNextStep)
        yield generation


def CreateGzipOfIndices(gzip_filepath, grammar_filepath):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    vocabSize = vocabulary.getMaxVocabularySize()

    PAD = vocabulary.indicesByTokens[vocabulary.padToken]
    START = vocabulary.indicesByTokens[vocabulary.startToken]
    END = vocabulary.indicesByTokens[vocabulary.endToken]

    f = gzip.open(gzip_filepath, 'rb')

    n_sentences = 0
    indices_list = []
    for line in f:
        n_sentences += 1
        sentence = line.strip().decode("utf-8")

        indices = np.array([PAD, START] + vocabulary.tokensToIndices(tokenize(sentence)) + [END])

        print(indices)
        indices_list.append(indices)

    # add number of lines in the file in the name of the file
    new_filepath = gzip_filepath[:-3] + '_lines' + str(n_sentences) + '_indices.gz'

    # save it to numpy
    indices_list = np.array(indices_list)
    np.save(new_filepath, indices_list)

    return new_filepath


def GzipToNextStepGenerator_new(gzip_filepath, grammar_filepath, batchSize, maxSentenceLen=None):
    # TODO: save the data in this format from the beginning, so people that want to test it don't have problems
    if not 'indices' in gzip_filepath:
        gzip_filepath = CreateGzipOfIndices(gzip_filepath, grammar_filepath)



def GzipToIndicesGenerator(gzip_filepath, grammar_filepath, batchSize):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)

    generatorSentences = generateFromGzip(
        gzipDatasetFilepath=gzip_filepath,
        batchSize=batchSize
    )
    generatorIndices = SentenceToIndicesGenerator(
        sentence_generator=generatorSentences,
        vocabulary=vocabulary
    )
    while True:
        generation = next(generatorIndices)
        yield generation


if __name__ == '__main__':
    grammar_filepath = '../data/simplerREBER_grammar.cfg'
    gzip_filepath = '../data/REBER_biased_train.gz'
    batchSize = 3
    generator = GzipToNextStepGenerator(gzip_filepath, grammar_filepath, batchSize)
    # check REBER generator

