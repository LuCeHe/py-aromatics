import os
from GenericTools.KerasTools.esoteric_tasks import ptb_reader
import numpy as np
import tensorflow as tf

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.join(*[CDIR, r'../data/ptb/'])


class PTBInput(object):
    """The input data."""

    def __init__(self, batch_size, num_steps, data, name=None):
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = ptb_reader.ptb_producer(data, batch_size, num_steps, name=name)
        self.input_generator = ptb_reader.ptb_producer_np_eprop(data, batch_size, num_steps, name=name)

def get_probabilities():
    batch_size = 1
    raw_data = ptb_reader.ptb_raw_data(DATAPATH, 'char')
    train_data, valid_data, test_data, vocab_size, word_to_id = raw_data
    num_steps = len(train_data) - 1
    data = PTBInput(batch_size=batch_size, num_steps=num_steps, data=train_data, name="TrainInput")

    x, y, done = next(data.input_generator)
    u, c = np.unique(x, return_counts=True)
    p = c / np.sum(c)

    print(p.shape)
    np.save(DATAPATH + '/prior_prob_ptb', p)
    return p, c, vocab_size

def test():
    p, c, vocab_size = get_probabilities()
    # plt.bar(u, p)
    # plt.show()
    # what is the bpc of

    sentence = np.array([[2., 2., 2., ], [2., 3., 13., ]])
    oh = tf.keras.utils.to_categorical(sentence, num_classes=vocab_size)
    print(oh)

    batch_size = sentence.shape[0]
    time_steps = sentence.shape[1]
    c_ = c[np.newaxis, np.newaxis, ...].astype(float)
    logits = np.repeat(np.repeat(c_, batch_size, axis=0), time_steps, axis=1)
    probabilities = logits/np.sum(c)
    labels = oh

    print('labels.shape: ', labels.shape)
    print('logits.shape: ', logits.shape)

    #xent = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, dim=-1, name=None)
    loss = tf.keras.losses.CategoricalCrossentropy()(oh, probabilities)
    mean_xent = loss
    bits_per_character = mean_xent / np.log(2)

    print(bits_per_character)

    sess = tf.Session()
    bpc = sess.run(bits_per_character)
    print(bpc)

if __name__ == '__main__':
    test()