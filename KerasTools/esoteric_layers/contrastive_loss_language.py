import tensorflow as tf


def contrastive_disorder(self, output_words, probs):
    time_steps = tf.shape(output_words)[1]
    half_time = tf.cast(time_steps / 2, tf.int32)
    splits = tf.split(output_words, [half_time, time_steps - half_time], axis=1)

    # splits = tf.split(original_sentences, 2, axis=1)
    disordered_sentences = tf.concat([splits[1], splits[0]], axis=1)
    # disordered_sentences = tf.argmax(disordered_sentences, axis=-1)
    # print(disordered_sentences.shape)
    # print(probs.shape)
    # cl_d = - self.coef_disorder * self.loss(disordered_sentences, probs)
    cl_d = - self.coef_disorder * tf.sigmoid(tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disordered_sentences, probs))
    self.add_loss(cl_d)
    self.add_metric(cl_d, name='contrastive_disorder', aggregation='mean')

def contrastive_random(self, probs):
    batch_size = tf.shape(probs)[0]
    seq_len = tf.shape(probs)[1]
    vocab_size = tf.shape(probs)[2]

    for i in range(self.n_random):
        p = tf.tile((1 / vocab_size)[None, None], [batch_size, vocab_size])
        random_words = tf.random.categorical(tf.math.log(p), seq_len)

        cl_r = - self.coef_random * tf.sigmoid(self.loss(random_words, probs))
        self.add_loss(cl_r)
        self.add_metric(cl_r, name='contrastive_random_{}'.format(i), aggregation='mean')

class ContrastiveLossLayer(tf.keras.layers.Layer):

    def __init__(self, coef_disorder=.1, coef_random=.1, n_random=1,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), **kwargs):
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
        self.disorder = lambda s, x, y: contrastive_disorder(s, x, y) if coef_disorder > 0 else None
        self.random = lambda s, x: contrastive_random(s, x) if coef_random > 0 else None

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

    def call(self, inputs, **kwargs):

        if isinstance(inputs, list) and len(inputs) == 2:
            output_words, probs = inputs
        else:
            output_words, probs = inputs, inputs

        self.disorder(self, output_words, probs)
        self.random(self, probs)

        return probs

    def get_config(self):
        config = {
            'coef_random': self.initial_coef_random,
            'coef_disorder': self.initial_coef_disorder,
            'loss': self.loss
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
