

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Median(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        if isinstance(inputs,list):
            inputs = tf.concat([tf.expand_dims(i, axis=-1) for i in inputs], axis=-1)
        median = tfp.stats.percentile(inputs, 50.0, interpolation='midpoint', axis=self.axis)
        return median


    def get_config(self):
        config = {
            'axis': self.axis,
        }
        return dict(list(super().get_config().items()) + list(config.items()))
