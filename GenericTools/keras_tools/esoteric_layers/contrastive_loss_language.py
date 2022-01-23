import tensorflow as tf
from functools import *

from GenericTools.keras_tools.convenience_operations import tf_shuffle_axis
from GenericTools.keras_tools.esoteric_losses import get_loss
from GenericTools.stay_organized.utils import str2val


def contrastive_disorder(self, y_true, y_pred, ):
    time_steps = tf.shape(y_true)[1]
    half_time = tf.cast(time_steps / 2, tf.int32)
    splits = tf.split(y_true, [half_time, time_steps - half_time], axis=1)

    # splits = tf.split(original_sentences, 2, axis=1)
    disordered_sentences = tf.concat([splits[1], splits[0]], axis=1)
    # disordered_sentences = tf.argmax(disordered_sentences, axis=-1)
    # print(disordered_sentences.shape)
    # print(probs.shape)
    # cl_d = - self.coef_disorder * self.loss(disordered_sentences, probs)
    contrastive_loss = - self.coef * tf.sigmoid(
        tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disordered_sentences, y_pred))
    self.add_loss(contrastive_loss)
    self.add_metric(contrastive_loss, name='contrastive_disorder', aggregation='mean')


def contrastive_random(self, y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    seq_len = tf.shape(y_pred)[1]
    vocab_size = tf.shape(y_pred)[2]

    for i in range(self.n_random):
        if self.categorical:
            p = tf.tile((1 / vocab_size)[None, None], [batch_size, vocab_size])
            random = tf.random.categorical(tf.math.log(p), seq_len)
        else:
            random = tf.random.normal(tf.shape(y_pred))

        contrastive_loss = - self.coef * tf.sigmoid(self.loss(random, y_pred))
        self.add_loss(contrastive_loss)
        self.add_metric(contrastive_loss, name='contrastive_random_{}'.format(i), aggregation='mean')


def axis_shuffle(self, y_true, y_pred, axis):
    s_true = tf_shuffle_axis(y_true, axis=axis, seed=None, name=None)

    contrastive_loss = - self.coef * tf.tanh(self.loss(s_true, y_pred))
    self.add_loss(contrastive_loss)
    self.add_metric(contrastive_loss, name='contrastive_inaxis', aggregation='mean')


def self_shuffle(self, y_true, y_pred, axis):
    sprobs = tf_shuffle_axis(y_pred, axis=axis)

    contrastive_loss = - self.coef * tf.tanh(tf.reduce_mean(tf.square(sprobs - y_pred)))
    self.add_loss(contrastive_loss)
    self.add_metric(contrastive_loss, name='selfcontrastive', aggregation='mean')


def contrastive_common(self, y_pred):
    most_common_words = tf.reduce_mean(tf.reduce_mean(y_pred, axis=0), axis=0)

    mean = tf.reduce_mean(most_common_words)
    silence_words = tf.cast(most_common_words > mean, tf.float32)[None, None]

    contrastive_loss = - self.coef * tf.tanh(tf.reduce_mean(tf.square(silence_words * y_pred)))
    self.add_loss(contrastive_loss)
    self.add_metric(contrastive_loss, name='contrastive_common', aggregation='mean')


# def promote_unlikely_words(self, y_true, y_pred):
#     most_common_words = tf.reduce_mean(tf.reduce_mean(y_true, axis=0), axis=0)
#
#     mean = tf.reduce_mean(most_common_words)
#     silence_words = tf.cast(most_common_words > mean, tf.float32)[None, None]
#
#     contrastive_loss = - self.coef * tf.tanh(tf.reduce_mean(tf.square(silence_words * y_pred)))
#     self.add_loss(contrastive_loss)
#     self.add_metric(contrastive_loss, name='contrastive_common', aggregation='mean')


class ContrastiveLossLayer(tf.keras.layers.Layer):

    def __init__(self, n_random=1,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), categorical=True, string_config='', **kwargs):
        super().__init__(**kwargs)
        self.n_random = n_random
        self.string_config = string_config
        self.categorical = categorical
        loss = get_loss(loss) if isinstance(loss, str) else loss

        if hasattr(loss, 'name'):
            loss.name = loss.name
        elif hasattr(loss, '__name__'):
            loss.name = loss.__name__
        else:
            raise NotImplementedError

        self.loss = loss
        self.coef = str2val(string_config, 'coefcontrastive', output_type=float, default=1)

        self.contrastives = []

        if 'disordercontrastive' in string_config:
            self.contrastives.append(contrastive_disorder)

        if 'randomcontrastive' in string_config:
            self.contrastives.append(contrastive_random)

        # in batch contrastive is equivalent to inaxis with axis=0
        # variation of the idea from https://arxiv.org/pdf/2004.13637.pdf
        if 'inaxiscontrastive' in string_config:
            axis = str2val(string_config, 'inaxiscontrastive', output_type=int)
            axis = [axis] if not isinstance(axis, list) else axis
            for ax in axis:
                c = partial(axis_shuffle, axis=ax)
                self.contrastives.append(c)

        # variation of the idea from https://arxiv.org/pdf/2004.13637.pdf
        if 'selfcontrastive' in string_config:
            axis = str2val(string_config, 'selfcontrastive', output_type=int)
            axis = [axis] if not isinstance(axis, list) else axis
            for ax in axis:
                c = partial(self_shuffle, axis=ax)
                self.contrastives.append(c)

        c = lambda s, t, p: contrastive_common(s, p) \
            if 'contrastivecommon' in string_config else None
        self.contrastives.append(c)

    def build(self, input_shape):
        self.coef = self.add_weight(name='contrastivecoef',
                                    shape=(),
                                    initializer=tf.keras.initializers.Constant(self.coef),
                                    trainable=False)

        self.built = True

    def call(self, inputs, **kwargs):

        if isinstance(inputs, list) and len(inputs) == 2:
            output_words, probs = inputs
        else:
            output_words, probs = inputs, inputs

        for c in self.contrastives:
            c(self, output_words, probs)

        return probs

    def get_config(self):
        config = {
            'string_config': self.string_config,
            'loss': self.loss,
            'n_random': self.n_random
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    t = tf.random.uniform((2, 3, 4))
    sentences = tf.argmax(t, axis=-1)

    c = ContrastiveLossLayer(string_config='inaxiscontrastive:0')
    input_t = tf.keras.layers.Input(t.shape[1:])
    input_s = tf.keras.layers.Input(sentences.shape[1:])
    out = c([input_s, input_t])
    model = tf.keras.models.Model([input_t, input_s], out)
    model.summary()
    model.compile('SGD', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit([t, sentences], sentences, epochs=2)
    # model.fit(t, t, epochs=2)
