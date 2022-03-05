import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy

from GenericTools.stay_organized.utils import rename


def sparse_categorical_accuracy_last(y_true, y_pred):
    y_true, y_pred = y_true[:, -1], y_pred[:, -1, :]
    y_true  = tf.expand_dims(y_true, 1)
    y_pred  = tf.expand_dims(y_pred, 1)
    return sparse_categorical_accuracy(y_true, y_pred)


def salchannel(channel):
    @rename('salchannel{}'.format(channel))
    def salc(y_true, y_pred):
        yt = y_true[..., channel]
        yp = y_pred[..., channel,:]
        return sparse_categorical_accuracy_last(yt, yp)

    return salc