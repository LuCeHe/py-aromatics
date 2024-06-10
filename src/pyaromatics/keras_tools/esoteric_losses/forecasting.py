import tensorflow as tf
from keras import backend as K

from pyaromatics.stay_organized.utils import rename


def mase(y_true, y_pred):
    sust = tf.reduce_mean(tf.abs(y_true[:, 1:] - y_true[:, :-1]))
    diff = tf.reduce_mean(tf.abs(y_pred - y_true))

    return diff / sust


def msae(y_true, y_pred):
    bern = tf.cast(tf.random.uniform(shape=tf.shape(y_true), minval=0, maxval=1) > .5, tf.float32)
    mse = tf.keras.losses.MSE(bern * y_true, bern * y_pred)
    mae = tf.keras.losses.MAE((1 - bern) * y_true, (1 - bern) * y_pred)
    loss = mse + mae
    return loss


def mase2(y_true, y_pred):
    M = tf.reduce_max(y_true)
    m = tf.reduce_min(y_true)
    sust = tf.abs(M - m)
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


def owa2(y_true, y_pred):
    return mase2(y_true, y_pred) + smape_loss(y_true, y_pred)


def mix_reg(y_true, y_pred):
    mse = tf.keras.losses.MSE(y_true, y_pred)
    mae = tf.keras.losses.MAE(y_true, y_pred)
    loss = mse / 4 + mae / 4 + owa2(y_true, y_pred) / 4
    return loss


def match_directions_rate(y_true, y_pred):
    # print('and here')
    st = tf.math.sign(y_true)
    sp = tf.math.sign(y_pred)
    equals = tf.cast(tf.math.equal(st, sp), tf.float32)
    return tf.reduce_mean(equals)


def drchannel(channel):
    @rename('drc{}'.format(channel))
    def drc(y_true, y_pred):
        yt = y_true[..., channel]
        yp = y_pred[..., channel]
        return match_directions_rate(yt, yp)

    return drc


if __name__ == '__main__':
    t = tf.random.normal((2, 3))
    p = tf.random.normal((2, 3))

    l = match_directions_rate(t, p)
    print(l)
