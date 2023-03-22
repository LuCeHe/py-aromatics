import math
import tensorflow as tf


def Guderman_T(T=1.):
    def activation(x):
        return x * (1 / 2 + 2 / math.pi * tf.math.atan(tf.tanh(x / T)))

    return activation


def GeLUnew_T(T=1.):
    def activation(x):
        return 0.5 * x * (1.0 + tf.math.tanh(math.sqrt(2.0 / math.pi) * (x / T + 0.044715 * tf.math.pow(x / T, 3.0))))

    return activation


def Swish_T(T=1.):
    def activation(x):
        return x * tf.math.sigmoid(x / T)

    return activation


def mish_sigmoid(x):
    return tf.tanh(tf.math.log(1 + tf.exp(x)))


def MISH(T=1.):
    def activation(x):
        return x * mish_sigmoid(x / T)

    return activation


def GumbelLU(T=1.):
    def activation(x):
        return x * tf.exp(-tf.exp(x / T))

    return activation


def mish_softmax(x):
    x = MISH(1.)(x)
    sum = tf.reduce_sum(x, axis=-1, keepdims=True)
    return x / sum


def relu_softmax(x):
    x = tf.nn.relu(x)
    sum = tf.reduce_sum(x, axis=-1, keepdims=True)
    return x / sum


def euclidean_softmax(x):
    x = tf.math.exp(x)
    sum = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    return x / tf.sqrt(sum)


activations_with_temperature = {
    'cguderman1': Guderman_T(),
    'cguderman.1': Guderman_T(.1),
    'cguderman.01': Guderman_T(.01),
    'swish1': Swish_T(),
    'swish.1': Swish_T(.1),
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

smooth_relus = {
    'gudermanlu': Guderman_T(),
    'gudermanlu.1': Guderman_T(.1),
    'swish.1': Swish_T(.1),
    'swish': Swish_T(1.),
    'mish': MISH(1.),
    'mish.1': MISH(.1),
    'gumbellu': GumbelLU(1.),
    'gumbellu.1': GumbelLU(.1),
    **activations_with_temperature
}

smooth_heavisides = {
    'sigmoid': tf.nn.sigmoid,
    'mish': mish_sigmoid,
}

softmaxes = {
    'softmax': tf.nn.softmax,
    'mishsoftmax': mish_softmax,
    'sigmoid': tf.nn.sigmoid,
    'relusoftmax': relu_softmax,
    'euclideansoftmax': euclidean_softmax
}
