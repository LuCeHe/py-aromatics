# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input pipeline for the listops dataset."""

import os
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

AUTOTUNE = tf.data.experimental.AUTOTUNE

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', '..', 'data', 'lra', 'listops'))
os.makedirs(DATADIR, exist_ok=True)


def rename_close_brackets(x):
    source = x['Source']
    source = tf.strings.regex_replace(source, ']', 'X')
    source = tf.strings.regex_replace(source, r'\(', '')
    source = tf.strings.regex_replace(source, r'\)', '')
    return {'Source': source, 'Target': x['Target']}


def preprocess_dataset(file_path, batch_size):
    """Preprocess dataset."""
    tf.logging.info(file_path)
    sel_cols = ['Source', 'Target']
    col_defaults = [tf.string, tf.int32]
    ds = tf.data.experimental.make_csv_dataset([file_path],
                                               batch_size,
                                               column_defaults=col_defaults,
                                               select_columns=sel_cols,
                                               field_delim='\t',
                                               header=True,
                                               num_epochs=1)
    ds = ds.unbatch()
    # we rename close brackets to X for this particular task because
    # tokenizer removes non alphanumeric.
    # since there is no trivial way to change this behaviour
    # we opt for an equivalent fix since the vocab in listops is fixed.
    ds = ds.map(rename_close_brackets, num_parallel_calls=AUTOTUNE)
    return ds


def get_datasets(n_devices,
                 task_name,
                 data_dir=None,
                 batch_size=256,
                 max_length=2000):
    """Get algorithmic datasets."""
    if batch_size % n_devices:
        raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                         (batch_size, n_devices))

    train_path = os.path.join(DATADIR, 'listops_train.tsv')
    val_path = os.path.join(DATADIR, 'listops_val.tsv')
    test_path = os.path.join(DATADIR, 'listops_test.tsv')

    train_dataset = preprocess_dataset(train_path, batch_size)
    val_dataset = preprocess_dataset(val_path, batch_size)
    test_dataset = preprocess_dataset(test_path, batch_size)

    n_train_samples = len(list(train_dataset))
    n_val_samples = len(list(val_dataset))
    n_test_samples = len(list(test_dataset))

    tf.logging.info('Finished preprocessing')
    tf.logging.info('Building vocab')
    # build vocab
    vocab_set = set()
    tokenizer = text.WhitespaceTokenizer()

    lengths = []
    for i, data in enumerate(val_dataset):
        examples = data['Source']
        examples = tokenizer.tokenize(examples.numpy())
        examples = np.reshape(examples, (-1)).tolist()
        lengths.append(len(examples))
        vocab_set.update(examples)
        if i % 1000 == 0:
            tf.logging.info('Processed {}'.format(i))
        if i > 1000:
            break
    vocab_set = list(set(vocab_set))
    tf.logging.info('Finished processing vocab size={}'.format(len(vocab_set)))

    encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

    def tf_encode(x):
        result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())),
                                [x, ],
                                tf.int32)
        result.set_shape([None])
        return result

    def tokenize(d):
        return {'inputs': tf_encode(d['Source'])[:max_length],
                'targets': d['Target']}

    train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

    max_shape = {'inputs': [max_length], 'targets': []}
    train_dataset = train_dataset.shuffle(
        buffer_size=1024, reshuffle_each_iteration=True).padded_batch(
        batch_size, padded_shapes=max_shape)
    val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=max_shape)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'n_train_samples': n_train_samples,
        'n_val_samples': n_val_samples,
        'n_test_samples': n_test_samples,
        'vocab_size': encoder.vocab_size
    }


def test():
    import time
    """Test."""
    start_time = time.time()
    train_ds, val_ds, test_ds, n_train_samples, vocab_size = get_datasets(8, 'listops', batch_size=256)
    print('time taken', time.time() - start_time)

    # batch_size = 256 -> steps_per_epoch = 96000

    # iter_train = iter(train_ds)
    # print(len(iter_train))
    # print(len(list(train_ds)))
    # print(len(list(iter_train)) )
    batch = next(iter(train_ds))
    print('vocab_size', vocab_size)
    print('batch', batch)


if __name__ == '__main__':
    test()
