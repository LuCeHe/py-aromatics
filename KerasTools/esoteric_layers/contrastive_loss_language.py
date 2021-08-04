import tensorflow as tf


class ContrastiveLossLayer(tf.keras.layers.Layer):

    def __init__(self, coef_disorder=.1, coef_random=.1, n_random=1,
                 loss=tf.keras.losses.CategoricalCrossentropy(), **kwargs):
        super().__init__(**kwargs)
        self.initial_coef_disorder = coef_disorder
        self.initial_coef_random = coef_random
        self.n_random = n_random

        if hasattr(loss, 'name'):
            loss.name = loss.name
        elif hasattr(loss, '__name__'):
            loss.name = loss.__name__
        else:
            raise NotImplementedError

        self.loss = loss

    def build(self, input_shape):
        self.coef_disorder = self.add_weight(name='coef_disorder',
                                             shape=(),
                                             initializer=tf.keras.initializers.Constant(self.initial_coef_disorder),
                                             trainable=False)

        self.coef_random = self.add_weight(name='coef_random',
                                           shape=(),
                                           initializer=tf.keras.initializers.Constant(self.initial_coef_random),
                                           trainable=False)

        self.built = True

    def call(self, inputs, training=None):
        output_words, probs = inputs

        batch_size = tf.shape(probs)[0]
        seq_len = tf.shape(probs)[1]
        vocab_size = tf.shape(probs)[2]

        if self.coef_disorder > 0:
            if 'categorical' in self.loss.name:
                input_words = output_words #tf.one_hot(tf.cast(tf.squeeze(input_words, axis=-1), tf.int32), vocab_size)
                original_sentences = tf.cast(input_words, dtype=tf.float32)
            else:
                original_sentences = output_words #input_words

            time_steps = tf.shape(original_sentences)[1]
            half_time = tf.cast(time_steps / 2, tf.int32)
            splits = tf.split(original_sentences, [half_time, time_steps - half_time], axis=1)

            # splits = tf.split(original_sentences, 2, axis=1)
            disordered_sentences = tf.concat([splits[1], splits[0]], axis=1)

            cl_d = - self.coef_disorder * self.loss(disordered_sentences, probs)
            self.add_loss(cl_d)
            self.add_metric(cl_d, name='contrastive_disorder', aggregation='mean')

        if self.coef_random > 0:
            for i in range(self.n_random):
                if 'categorical' in self.loss.name:
                    p = tf.tile((1 / vocab_size)[None, None], [batch_size, vocab_size])
                    ps = tf.random.categorical(tf.math.log(p), seq_len)
                    random_words = tf.one_hot(ps, vocab_size)
                else:
                    std = tf.math.reduce_std(output_words)
                    random_words = std * tf.random.normal(shape=tf.shape(probs))

                cl_r = - self.coef_random * self.loss(random_words, probs)
                self.add_loss(cl_r)
                self.add_metric(cl_r, name='contrastive_random_{}'.format(i), aggregation='mean')

        return probs

    def get_config(self):
        config = {
            'coef_random': self.initial_coef_random,
            'coef_disorder': self.initial_coef_disorder,
            'loss': self.loss
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
