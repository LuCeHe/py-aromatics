import tensorflow as tf


class DropIn(tf.keras.layers.Layer):

    def __init__(self, drop_prob, binary=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.binary = binary

        if binary:
            self.threshold = lambda x: tf.clip_by_value(x, -.1, 1.)
        else:
            self.threshold = lambda x: x

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        to_add = tf.random.stateless_binomial(shape=tf.shape(inputs), seed=[123, 456], counts=1, probs=self.drop_prob)
        to_add = tf.cast(to_add, tf.float32)

        droppedin = self.threshold(inputs + to_add)

        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)
        dropped_words = is_train * droppedin + (1 - is_train) * inputs
        return dropped_words

    def get_config(self):
        config = {
            'drop_prob': self.dropword_prob,
            'binary': self.vocab_size,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    probs = .2
    a = tf.random.stateless_binomial(
        shape=[2, 10], seed=[123, 456], counts=1, probs=probs)

    t = tf.random.uniform((2, 3,))
    print(a)

    layer = DropIn(.2, binary=True)(t, training=True)
    print(t)
    print(layer)
