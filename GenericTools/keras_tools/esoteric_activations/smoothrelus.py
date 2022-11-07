import math

import tensorflow as tf


def Guderman_T(T=1.):
    def guderT(x):
        return x * (1 / 2 + 2 / math.pi * tf.math.atan(tf.tanh(x / T)))

    return guderT


def GeLUnew_T(T=1.):
    def gelunT(x):
        return 0.5 * x * (1.0 + tf.math.tanh(math.sqrt(2.0 / math.pi) * (x / T + 0.044715 * tf.math.pow(x / T, 3.0))))

    return gelunT


def Swish_T(T=1.):
    def swishT(x):
        return x * tf.math.sigmoid(x / T)

    return swishT


activations_with_temperature = {
    'cguderman1': Guderman_T(),
    'cguderman.1': Guderman_T(.1),
    'cguderman.01': Guderman_T(.01),
    'cswish1': Swish_T(),
    'cswish.1': Swish_T(.1),
    'cswish.01': Swish_T(.01),
    'relu': tf.nn.relu,
    'gelu_new': GeLUnew_T()
}

critical_cws = {
    'cguderman1': 1.990,
    'cguderman.1': 1.990,
    'cguderman.01': 1.990,
    'cswish1': 1.988,
    'cswish.1': 1.988,
    'cswish.01': 1.988,
    'gelu_new': 1.983,
    'relu': 2.
}

critical_cbs = {
    'cguderman1': 0.103,
    'cguderman.1': 0.103 * .1 ** 2,
    'cguderman.01': 0.103 * .01 ** 2,
    'cswish1': 0.555,
    'cswish.1': 0.555 * .1 ** 2,
    'cswish.01': 0.555 * .01 ** 2,
    'gelu_new': 0.173,
    'relu': 0.
}
