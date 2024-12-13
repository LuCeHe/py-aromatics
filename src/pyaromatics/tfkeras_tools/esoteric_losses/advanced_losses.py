from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance
# import tensorflow_addons as tfa

from pyaromatics.keras_tools.esoteric_losses.forecasting import smape_loss

"""
sources:
https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163
"""


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


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


def sparse_mode_accuracy(y_true, y_pred):
    # print(y_true)
    # print(y_pred.shape, y_true.shape)
    y_pred = tf.cast(tf.argmax(tf.reduce_mean(y_pred, axis=1), axis=1), tf.float32)
    y_true = tf.cast(tf.reduce_mean(y_true, axis=1), tf.float32)
    # print(y_pred.shape, y_true.shape)
    equal = tf.cast(tf.math.equal(y_pred, y_true), tf.float32)
    acc = tf.reduce_mean(equal)
    return acc

def second_half(tensor):
    time_steps = tf.shape(tensor)[1]
    half_time = tf.cast(time_steps / 2, tf.int32)
    splits = tf.split(tensor, [half_time, time_steps - half_time], axis=1)
    return splits[1]

def second_half_mode_accuracy(y_true, y_pred):
    y_true = second_half(y_true)
    y_pred = second_half(y_pred)
    true = tf.cast(tf.argmax(tf.reduce_sum(y_true, axis=1), axis=1), tf.float32)
    pred = tf.cast(tf.argmax(tf.reduce_sum(y_pred, axis=1), axis=1), tf.float32)
    equal = tf.cast(tf.math.equal(pred, true), tf.float32)
    acc = tf.reduce_mean(equal)
    return acc


def sparse_last_time_accuracy(y_true, y_pred):
    true = tf.cast(y_true[:, -1], tf.float32)
    pred = tf.cast(tf.argmax(y_pred[:, -1, :], axis=1), tf.float32)
    equal = tf.cast(tf.math.equal(pred, true), tf.float32)
    acc = tf.reduce_mean(equal)
    return acc


def zeros_categorical_accuracy(y_true, y_pred):
    n_tot = tf.reduce_sum(y_true, axis=[1, 2])
    depth = tf.shape(y_true)[-1]
    new_t = y_true * tf.one_hot(tf.math.argmax(y_pred, axis=2), depth=depth)
    n_right = tf.reduce_sum(new_t, axis=[1, 2])
    return n_right / n_tot


def oh_bpc(y_true, y_pred):
    mean_xent = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    bits_per_character = mean_xent / tf.math.log(2.)
    return bits_per_character


def bpc(y_true, y_pred):
    mean_xent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    bits_per_character = mean_xent / tf.math.log(2.)
    return bits_per_character


def bpc_2(y_true, y_pred):
    bits_per_character = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[2]) / tf.math.log(2.)
    bits_per_character = tf.math.reduce_mean(tf.math.reduce_mean(bits_per_character, axis=1), axis=0)
    # bits_per_character = tf.math.reduce_mean(bits_per_character)
    return bits_per_character


def entropy_data(y_true, y_pred):
    bits_per_character = -tf.reduce_sum(y_true * tf.math.log(y_true + 1e-8), axis=[2]) / tf.math.log(2.)
    bits_per_character = tf.math.reduce_mean(tf.math.reduce_mean(bits_per_character, axis=1), axis=0)
    return bits_per_character


def entropy_model(y_true, y_pred):
    bits_per_character = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=[2]) / tf.math.log(2.)
    bits_per_character = tf.math.reduce_mean(tf.math.reduce_mean(bits_per_character, axis=1), axis=0)
    return bits_per_character


def bound_a(y_true, y_pred):
    mse = tf.keras.losses.MSE(y_true, y_pred)
    bound = (mse - 1) / 2
    return bound


def bound_b(y_true, y_pred):
    mean_xent = tf.keras.losses.CategoricalCrossentropy()(y_pred, y_true)
    mse = tf.keras.losses.MSE(y_true, y_pred)
    bound = mse - mean_xent
    return bound


def bound_c(y_true, y_pred):
    mean_xent = tf.keras.losses.CategoricalCrossentropy()(y_true, y_true)
    mse = tf.keras.losses.MSE(y_true, y_pred)
    bound = mse - mean_xent
    return bound


def oh_perplexity(y_true, y_pred):
    mean_xent = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    p = tf.exp(mean_xent)
    return p


def perplexity(y_true, y_pred):
    mean_xent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    p = tf.exp(mean_xent)
    return p


def masked_sparse_crossentropy(mask_value):
    def masked_xent(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.equal(y_true, mask_value)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        mloss = loss * mask
        # take average w.r.t. the number of unmasked entries
        return K.sum(mloss) / K.sum(mask)

    return masked_xent


def sparse_perplexity(y_true, y_pred):
    mean_xent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    p = tf.exp(mean_xent)
    return p


def sparsesmape(y_true, y_pred):
    vocab_size = y_pred.shape[-1]
    oh_true = tf.one_hot(tf.cast(y_true, tf.int32), vocab_size)
    loss = smape_loss(oh_true, y_pred)
    return loss


def masked_sparse_perplexity(mask_value):
    mxent = masked_sparse_crossentropy(mask_value)

    def masked_perplexity(y_true, y_pred):
        mean_xent = mxent(y_true, y_pred)
        p = tf.exp(mean_xent)
        return p

    return masked_perplexity

masked_perplexity = masked_sparse_perplexity

def f1(y_true, y_pred):
    actual, predicted = y_true, y_pred
    # https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
    # Now if you have your actual and predicted values as vectors of 0/1, you can calculate TP, TN, FP, FN using tf.count_nonzero:
    TP = tf.math.count_nonzero(predicted * actual)
    # TN = tf.math.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.math.count_nonzero(predicted * (actual - 1))
    FN = tf.math.count_nonzero((predicted - 1) * actual)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def sparse_f1_on_max(y_true, y_pred):
    num_classes = y_pred.shape[-1]

    max_pred = tf.argmax(y_pred, -1)
    y_true = tf.cast(y_true, tf.int32)
    oh_true = tf.one_hot(y_true, depth=num_classes)
    oh_pred = tf.one_hot(max_pred, depth=num_classes)
    return f1(oh_true, oh_pred)


def masked_f1_on_max(num_classes, mask_value):
    def masked_f1_on_max(y_true, y_pred):
        mask = 1 - tf.cast(K.equal(y_true, mask_value), tf.float32)
        mask = tf.expand_dims(mask, -1)

        max_pred = tf.argmax(y_pred, -1)
        y_true = tf.cast(y_true, tf.int32)
        oh_true = tf.one_hot(y_true, depth=num_classes)
        oh_pred = tf.one_hot(max_pred, depth=num_classes)

        TP = tf.math.count_nonzero(oh_pred * oh_true * mask)
        FP = tf.math.count_nonzero(oh_pred * (oh_true - 1) * mask)
        FN = tf.math.count_nonzero((oh_pred - 1) * oh_true * mask)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    return masked_f1_on_max


def sparse_f1(num_classes):
    def score(y_true, y_pred):
        # num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        f1 = tfa.metrics.F1Score(num_classes=num_classes)
        result = f1.update_state(y_true, y_pred)
        return result.result()

    return score


def si_sdr_loss(y_true, y_pred):
    # print("######## SI-SDR LOSS ########")
    x = tf.cast(y_true, tf.float32)
    y = tf.cast(y_pred, tf.float32)

    # x = tf.squeeze(y_true, axis=-1)
    # y = tf.squeeze(y_pred, axis=-1)
    smallVal = 1e-9  # To avoid divide by zero
    a = tf.reduce_sum(y * x, axis=1, keepdims=True) / (tf.reduce_sum(x * x, axis=1, keepdims=True) + smallVal)

    xa = a * x
    xay = xa - y
    d = tf.reduce_sum(xa * xa, axis=1, keepdims=True) / (tf.reduce_sum(xay * xay, axis=1, keepdims=True) + smallVal)
    # d1=tf.zeros(d.shape)
    d1 = d == 0
    d1 = 1 - tf.cast(d1, tf.float32)

    d = -tf.reduce_mean(10 * d1 * log10(d + smallVal))
    return d


def pearson_r(y_true, y_pred):
    epsilon= 1e-8
    # original: https://github.com/WenYanger/Keras_Metrics/
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)+epsilon
    r = r_num / r_den
    return K.mean(r)


def pearson_r_a(y_true, y_pred):
    # original: https://github.com/WenYanger/Keras_Metrics/
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def pearson_loss(y_true, y_pred):
    return - pearson_r_a(y_true, y_pred)


@tf.custom_gradient
def clip_value_no_grad(x):
    y = tf.clip_by_value(x, -2, 2)

    def custom_grad(dy):
        return dy

    return y, custom_grad


def well_loss(min_value=-120, max_value=40, walls_type='clip_relu_no_clip_grad', axis='all'):
    def wloss(x):
        if walls_type == 'sigmoid':
            loss = -tf.math.sigmoid(x - min_value) + tf.math.sigmoid(x - max_value)
        elif walls_type == 'relu':
            loss = tf.nn.relu(-x + min_value) + tf.nn.relu(x - max_value)
        elif walls_type == 'clip_relu_no_clip_grad':
            loss = tf.nn.relu(-x + min_value) + tf.nn.relu(x - max_value)
            loss = clip_value_no_grad(loss)
        elif walls_type == 'squared':
            loss = tf.square(tf.nn.relu(x - max_value)) + tf.square(tf.nn.relu(min_value - x))
        else:
            raise NotImplementedError

        if axis == 'all':
            return tf.reduce_mean(loss)
        elif axis == None:
            return loss
        else:
            return tf.reduce_mean(loss, axis=axis)

    return wloss


def test_well():
    import matplotlib.pyplot as plt
    x = np.linspace(-1, 2, 100)
    for well_shape in ['sigmoid', 'relu', 'squared']:
        w = well_loss(min_value=1, max_value=1, walls_type=well_shape, axis=None)(x)
        plt.plot(x, w, label=well_shape)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_well()
