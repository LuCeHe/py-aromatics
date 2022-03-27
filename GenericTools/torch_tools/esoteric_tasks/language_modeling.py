# snippets from
# https://github.com/dnouri/pytorch-examples/blob/master/word_language_model/main.py

import os, torch, pickle
from torchtext import datasets


def download_ptb(datadir):
    ptb = datasets.PennTreebank(datadir)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, trainpath, valpath, testpath):
        self.dictionary = Dictionary()
        self.train = self.tokenize(trainpath)
        self.valid = self.tokenize(valpath)
        self.test = self.tokenize(testpath)

    def tokenize(self, path):
        """Tokenizes a text file."""

        if not os.path.exists(path):
            path = str(path).replace('PennTreebank/', '')

        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    data = data.cuda() if torch.cuda.is_available() else data
    return data, nbatch


def get_batch(source, bptt, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i * bptt:i * bptt + seq_len].T
    target = source[i * bptt + 1:i * bptt + 1 + seq_len].T  # .view(-1)
    return data, target[..., -1]


class PennTreeBankTask():
    def __init__(self, batch_size, epochs, steps_per_epoch, maxlen, train_val_test, datadir=None, string_config=''):
        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            train_val_test=train_val_test,
            string_config=string_config)

        if datadir is None:
            raise ValueError('Define where you want the data to be saved with the argument datadir.')


        PTBDIR = os.path.join(datadir, 'PennTreebank')
        os.makedirs(PTBDIR)

        pickle_name = os.path.join(PTBDIR, 'corpus.pickle')
        if not os.path.exists(pickle_name):
            download_ptb(datadir)
            corpus = Corpus(*[os.path.join(PTBDIR, 'ptb.{}.txt'.format(set)) for set in ['train', 'valid', 'test']])
            with open(pickle_name, 'wb') as handle:
                pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(pickle_name, 'rb') as handle:
                corpus = pickle.load(handle)

        if train_val_test == 'train':
            self.data, n_words = batchify(corpus.train, batch_size)
        elif train_val_test in ['valid', 'validation', 'val']:
            self.data, n_words = batchify(corpus.valid, batch_size)
        elif train_val_test == 'test':
            self.data, n_words = batchify(corpus.test, batch_size)
        else:
            raise NotImplementedError

        del corpus

        self.steps_per_epoch = n_words // (maxlen * batch_size) if steps_per_epoch == None else steps_per_epoch
        self.in_channels = 1
        self.out_channels = 10000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        data, targets = get_batch(self.data, self.maxlen, index)
        return data.to(self.device), targets[..., None].long().to(self.device)
