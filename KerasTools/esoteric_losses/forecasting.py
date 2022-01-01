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
