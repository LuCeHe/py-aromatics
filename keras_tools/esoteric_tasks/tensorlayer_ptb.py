import os
import numpy as np

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'ptb'))


# original: https://github.com/tensorlayer/TensorLayer/blob/master/tensorlayer/iterate.py
def ptb_iterator(raw_data, batch_size, num_steps):
    """Generate a generator that iterates on a list of words, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py>`__.
    Yields the source contexts and the target context by the given batch_size and num_steps (sequence_length).
    In TensorFlow's tutorial, this generates `batch_size` pointers into the raw
    PTB data, and allows minibatch iteration along these pointers.
    Parameters
    ----------
    raw_data : a list
            the context in list format; note that context usually be
            represented by splitting by space, and then convert to unique
            word IDs.
    batch_size : int
            the batch size.
    num_steps : int
            the number of unrolls. i.e. sequence_length
    Yields
    ------
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
    Raises
    ------
    ValueError : if batch_size or num_steps are too high.
    Examples
    --------
    >>> train_data = [i for i in range(20)]
    >>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
    >>>     x, y = batch
    >>>     print(x, y)
    ... [[ 0  1  2] <---x                       1st subset/ iteration
    ... [10 11 12]]
    ... [[ 1  2  3] <---y
    ... [11 12 13]]
    ...
    ... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
    ... [13 14 15]] <--- 2nd batch input
    ... [[ 4  5  6]  <--- 1st batch target
    ... [14 15 16]] <--- 2nd batch target
    ...
    ... [[ 6  7  8]                             3rd subset/ iteration
    ... [16 17 18]]
    ... [[ 7  8  9]
    ... [17 18 19]]
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y, i+1)


if __name__ == '__main__':
    DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'ptb', 'simple-examples', 'data'))
    train_path = os.path.join(DATAPATH, 'ptb.train.txt')
    # word_to_id =

    with open(train_path) as f:
        contents = f.read()

    # contents = contents.replace('\n', '').replace('  ', ' ')
    list_words = [w for w in contents.split(' ') if not w is '']
    vocabulary, counts = np.unique(list_words, return_counts=True)
    word_to_id = {v: i for i, v in enumerate(vocabulary)}
    id_to_word= {i: v for i, v in enumerate(vocabulary)}
    ptb_numbers = [word_to_id[w] for w in list_words]


    batch_size, num_steps = 2, 10
    ptb_it = ptb_iterator(ptb_numbers, batch_size, num_steps)
    input_batch, output_batch, _ = next(ptb_it)
    print(input_batch.shape, output_batch.shape)

    print([id_to_word[i] for i in input_batch[0]])
    print([id_to_word[i] for i in output_batch[0]])