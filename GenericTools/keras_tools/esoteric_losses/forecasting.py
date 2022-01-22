import tensorflow as tf
from tensorflow.keras import backend as K


def mase(y_true, y_pred):
    sust = tf.reduce_mean(tf.abs(y_true[:, 1:] - y_true[:, :-1]))
    diff = tf.reduce_mean(tf.abs(y_pred - y_true))

    return diff / sust


def smape_loss(y_true, y_pred):
    # by PigSpdr:
    # https://datascience.stackexchange.com/questions/41093/using-smape-as-a-loss-function-for-an-lstm
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape


def true_smape_loss(y_true, y_pred):
    # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    epsilon = 1e-6
    den = K.abs(y_true) + K.abs(y_pred) + epsilon
    smape = K.abs(y_pred - y_true) / den
    return tf.reduce_mean(smape)


def true_smape_loss_b(y_true, y_pred):
    # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    epsilon = 1e-6
    den = tf.reduce_mean(K.abs(y_true + y_pred) + epsilon)
    smape = K.abs(y_pred - y_true) / den
    return tf.reduce_mean(smape)


def smape_loss_b(y_true, y_pred):
    epsilon = 1e-6
    summ = K.abs(y_true) + K.abs(y_pred) + epsilon
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return tf.reduce_sum(smape)


def smape_loss_c(y_true, y_pred):
    epsilon = 1e-6
    summ = tf.square(y_true) + tf.square(y_pred) + epsilon
    smape = tf.square(y_pred - y_true) / summ * 2.0
    return tf.reduce_sum(smape)


def owa(y_true, y_pred):
    return mase(y_true, y_pred) + smape_loss(y_true, y_pred)


def match_directions_rate(y_true, y_pred):
    st = tf.math.sign(y_true)
    sp = tf.math.sign(y_pred)
    equals = tf.cast(tf.math.equal(st, sp), tf.float32)
    return tf.reduce_mean(equals)

if __name__ == '__main__':
    t = tf.random.normal((2,3))
    p = tf.random.normal((2,3))

    l = match_directions_rate(t,p)
    print(l)