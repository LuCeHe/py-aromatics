import tensorflow as tf


class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, skip_prob, **kwargs):
        super().__init__(**kwargs)
        self.skip_prob = skip_prob

    def call(self, inputs, training=None):
        layer_output, layer_input = inputs
        assert len(layer_input.shape) == len(layer_output.shape)
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        batch_size = tf.shape(layer_input)[0]
        # seq_len = tf.shape(inputs)[1]

        skip_prob = tf.cast(self.skip_prob, tf.float32)
        p = tf.tile([[skip_prob, 1 - skip_prob]], [batch_size, 1])
        mask = tf.cast(tf.random.categorical(tf.math.log(p), 1), dtype=tf.float32) #[..., None]

        choice = layer_output * mask + layer_input * (1 - mask)
        return choice

    def get_config(self):
        config = {
            'skip_prob': self.skip_prob,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
