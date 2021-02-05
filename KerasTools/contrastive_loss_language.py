import tensorflow as tf


class ContrastiveLossLayer(tf.keras.layers.Layer):

    def __init__(self, coef_disorder=.1, coef_random=.1, **kwargs):
        super().__init__(**kwargs)
        self.coef_disorder = coef_disorder
        self.coef_random = coef_random

    def call(self, inputs, training=None):
        input_words, probs = inputs

        batch_size = tf.shape(probs)[0]
        seq_len = tf.shape(probs)[1]
        vocab_size = tf.shape(probs)[2]

        if self.coef_disorder>0:
            input_words = tf.one_hot(tf.cast(tf.squeeze(input_words, axis=-1), tf.int32), vocab_size)

            original_sentences = tf.cast(input_words, dtype=tf.float32)
            splits = tf.split(original_sentences, 2, axis=1)
            disordered_sentences = tf.concat([splits[1], splits[0]], axis=1)

            cl_d = - self.coef_disorder * tf.keras.losses.CategoricalCrossentropy()(disordered_sentences, probs)
            self.add_loss(cl_d)
            self.add_metric(cl_d, name='contrastive_disorder', aggregation='mean')


        if self.coef_random>0:
            p = tf.tile((1 / vocab_size)[None, None], [batch_size, vocab_size])
            ps = tf.random.categorical(tf.math.log(p), seq_len)
            random_words = tf.one_hot(ps, vocab_size)

            cl_r = - self.coef_random * tf.keras.losses.CategoricalCrossentropy()(random_words, probs)
            self.add_loss(cl_r)
            self.add_metric(cl_r, name='contrastive_random', aggregation='mean')

        return probs

    def get_config(self):
        config = {
            'coef_random': self.coef_random,
            'coef_disorder': self.coef_disorder,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
