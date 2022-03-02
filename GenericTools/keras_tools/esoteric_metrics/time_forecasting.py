import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy

def sparse_categorical_accuracy_last(y_true, y_pred):
    y_true, y_pred = y_true[:, -1], y_pred[:, -1, :]
    y_true  = tf.expand_dims(y_true, 1)
    y_pred  = tf.expand_dims(y_pred, 1)
    return sparse_categorical_accuracy(y_true, y_pred)