# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os, json, sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

Py3 = sys.version_info[0] == 3


def _read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename, char_or_word):
    if not os.path.isfile(filename + '.dict'):
        length_document = 0
        space = ' ' if char_or_word == 'word' else '_'
        with tf.io.gfile.GFile(filename, "r") as f:
            words = []
            for line in tqdm(f):
                line_words = line.replace("\n", space).split()
                length_document += len(line_words)
                words += line_words
                words = np.unique(words).tolist()

        word_to_id = dict(zip(words, range(len(words))))

        json_dict = json.dumps(word_to_id)
        with open(filename + '.dict', "w") as f:
            f.write(json_dict)

        properties = {'length_document': length_document}
        json_dict = json.dumps(properties)
        with open(filename + '.props', "w") as f:
            f.write(json_dict)
    else:
        with open(filename + '.dict') as f:
            word_to_id = json.load(f)

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, char_or_word='word', name='ptb'):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    if char_or_word == 'word':
        train_path = os.path.join(data_path, "ptb.train.txt")
        valid_path = os.path.join(data_path, "ptb.valid.txt")
        test_path = os.path.join(data_path, "ptb.test.txt")
    elif char_or_word == 'char':
        train_path = os.path.join(data_path, "{}.char.train.txt".format(name))
        valid_path = os.path.join(data_path, "{}.char.valid.txt".format(name))
        test_path = os.path.join(data_path, "{}.char.test.txt".format(name))
    else:
        raise NotImplementedError

    word_to_id = _build_vocab(train_path, char_or_word=char_or_word)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.compat.v1.assert_positive(epoch_size,
                                                 message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        # x = tf.Print(x, [x], message="ptb_producer x", summarize=100)
        return x, y


def ptb_producer_np(raw_data, batch_size, num_steps, name=None):
    import numpy as np
    data_len = np.size(raw_data)
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    i = 0
    while True:
        x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
        x = x.reshape([batch_size, num_steps])
        y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]
        y = y.reshape([batch_size, num_steps])
        yield x, y
        i += 1
        if i == epoch_size:
            i = 0


def ptb_producer_np_eprop(raw_data, batch_size, num_steps):
    import numpy as np
    data_len = np.size(raw_data)
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    i = 0
    while True:
        x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
        x = x.reshape([batch_size, num_steps])
        y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]
        y = y.reshape([batch_size, num_steps])
        yield x, y, i + 1 == epoch_size
        i += 1
        if i == epoch_size:
            i = 0
