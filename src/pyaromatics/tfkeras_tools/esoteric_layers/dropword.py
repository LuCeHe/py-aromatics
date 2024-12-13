import tensorflow as tf


class DropWord(tf.keras.layers.Layer):

    def __init__(self, dropword_prob, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.dropword_prob = dropword_prob
        self.vocab_size = vocab_size

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        spike_dropout = tf.cast(self.dropword_prob, tf.float32)
        p = tf.tile([[spike_dropout, 1 - spike_dropout]], [batch_size, 1])
        mask = tf.cast(1 - tf.random.categorical(tf.math.log(p), seq_len), dtype=tf.float32)

        p = tf.tile([[1 / self.vocab_size] * self.vocab_size], [batch_size, 1])
        samples = tf.cast(tf.random.categorical(tf.math.log(p), seq_len), dtype=tf.float32)

        if len(tf.shape(inputs)) == 3:
            mask = mask[..., None]
            samples = samples[..., None]

        dropped_words = inputs * (1 - mask) + mask * samples
        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)

        dropped_words = is_train * dropped_words + (1 - is_train) * inputs
        # dropped_words = tf.cast(dropped_words, tf.int32)
        return dropped_words

    def get_config(self):
        config = {
            'dropword_prob': self.dropword_prob,
            'vocab_size': self.vocab_size,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
