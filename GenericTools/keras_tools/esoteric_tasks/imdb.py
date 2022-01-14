import os
import numpy as np
from string import punctuation
from collections import Counter
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import tensorflow.keras as keras
from GenericTools.stay_organized.download_utils import download_url

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))
vocab_filename = DATADIR + '/txt_sentoken/vocab.txt'

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/review_polarity.tar.gz'


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load all docs in a directory
def create_vocabulary(directory, vocab):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


# load all docs in a directory
def process_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

if not os.path.isfile(DATADIR + '/txt_sentoken/imdb.npy'):
    if not os.path.isfile(DATADIR + '/' + url.split('/')[-1]):
        download_url(url, DATADIR)
        # TODO: extract the tar automatically

    if not os.path.isfile(vocab_filename):
        # define vocab
        vocab = Counter()
        # add all docs to vocab
        create_vocabulary(DATADIR + '/txt_sentoken/pos', vocab)
        create_vocabulary(DATADIR + '/txt_sentoken/neg', vocab)
        # print the size of the vocab
        print('len(vocab): ', len(vocab))
        # print the top words in the vocab
        print(vocab)
        print(vocab.most_common(50))
        vocab = {tuple[0]: tuple[1] for tuple in vocab.most_common(10000)}

        # keep tokens with a min occurrence
        min_occurane = 2
        tokens = [k for k, c in vocab.items() if c >= min_occurane]
        print('len(tokens): ', len(tokens))

        # save tokens to a vocabulary file
        save_list(tokens, vocab_filename)

    # load the vocabulary
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    # load all training reviews
    positive_lines = process_docs(DATADIR + '/txt_sentoken/pos', vocab, True)
    negative_lines = process_docs(DATADIR + '/txt_sentoken/neg', vocab, True)

    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    docs = negative_lines + positive_lines
    tokenizer.fit_on_texts(docs)

    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
    print(Xtrain.shape)

    # load all test reviews
    positive_lines = process_docs(DATADIR + '/txt_sentoken/pos', vocab, False)
    negative_lines = process_docs(DATADIR + '/txt_sentoken/neg', vocab, False)
    docs = negative_lines + positive_lines
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
    print(Xtest.shape)

    ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
    ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

    # convert class vectors to binary class matrices
    ytrain = keras.utils.to_categorical(ytrain, 2)
    ytest = keras.utils.to_categorical(ytest, 2)

    print(np.max(Xtrain), np.min(Xtrain))

    data = {'Xtrain': Xtrain,
            'ytrain': ytrain,
            'Xtest': Xtest,
            'ytest': ytest,
            }

    np.save(DATADIR + '/txt_sentoken/imdb.npy',
            data)
