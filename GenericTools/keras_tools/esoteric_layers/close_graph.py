import tensorflow as tf

class CloseGraph(tf.keras.layers.Layer):
    """ Inspired by Deeply-Supervised Nets """

    def call(self, inputs, training=None):
        return inputs[0]
