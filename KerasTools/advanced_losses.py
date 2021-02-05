from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance

try:
    from segmentation_models.base import Loss
except:
    from segmentation_models.segmentation_models.base import Loss

"""
sources:
https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163
"""


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


class SurfaceLoss(Loss):

    def __init__(self):
        super().__init__(name='surface_loss')

    def __call__(self, gt, pr):
        return surface_loss(gt, pr)


class SNRLoss(Loss):

    def __init__(self, epsilon=1e-5):
        super().__init__(name='snr_loss')
        self.epsilon = epsilon

    def __call__(self, gt, pr):
        S = gt
        N = gt - pr + self.epsilon
        SNR = tf.math.truediv(S, N)
        return SNR


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou



def mode_accuracy(y_true, y_pred):
    true = tf.cast(tf.argmax(tf.reduce_sum(y_true, axis=1), axis=1), tf.float32)
    pred = tf.cast(tf.argmax(tf.reduce_sum(y_pred, axis=1), axis=1), tf.float32)
    equal = tf.cast(tf.math.equal(pred, true), tf.float32)
    acc = tf.reduce_mean(equal)
    return acc


def zeros_categorical_accuracy(y_true, y_pred):
    n_tot = tf.reduce_sum(y_true, axis=[1, 2])
    depth = tf.shape(y_true)[-1]
    new_t = y_true * tf.one_hot(tf.math.argmax(y_pred, axis=2), depth=depth)
    n_right = tf.reduce_sum(new_t, axis=[1, 2])
    return n_right / n_tot


def bpc(y_true, y_pred):
    mean_xent = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    bits_per_character = mean_xent / np.log(2)
    return bits_per_character

