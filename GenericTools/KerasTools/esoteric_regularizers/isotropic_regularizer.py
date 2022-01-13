import tensorflow as tf
import tensorflow.keras.backend as K



class Isotropic(tf.keras.regularizers.Regularizer):
    """
    inspired by https://arxiv.org/pdf/1702.01417.pdf
        encourage std of singular values to be zero
    """

    def __init__(self, coef=10.):  # pylint: disable=redefined-outer-name
        self.coef = K.cast_to_floatx(coef)

    def __call__(self, x):
        # encourage std of singular values to be zero
        s, _, _ = tf.linalg.svd(x)
        loss = tf.square(tf.math.reduce_std(s))
        return self.coef * loss

    def get_config(self):
        return {'coef': float(self.coef)}
