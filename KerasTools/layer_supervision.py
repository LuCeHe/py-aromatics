import tensorflow as tf
from tensorflow.keras.layers import *

class LayerSupervision(tf.keras.layers.Layer):
    """ Inspired by Deeply-Supervised Nets """

    def __init__(self, coef=.1, n_classes=2, kernel_size=16, loss=tf.keras.losses.CategoricalCrossentropy(), **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        self.loss = loss
        self.conv = Conv1D(n_classes, kernel_size, padding='causal')


    def call(self, inputs, training=None):
        representation, classes = inputs
        conv = self.conv(representation)
        loss = self.coef * self.loss(classes, conv)
        self.add_loss(loss)
        self.add_metric(loss, name='layer_supervision_{}'.format(self.name), aggregation='mean')

        return representation

    def get_config(self):
        config = {
            'coef': self.coef,
            'loss': self.loss
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
