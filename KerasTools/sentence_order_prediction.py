import tensorflow as tf

"""
source:
    - https://arxiv.org/pdf/1909.11942.pdf
"""


class SentenceOrderPrediction(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SentenceOrderPrediction, self).__init__(**kwargs)

    def call(self, inputs, training=None):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        batch_size = tf.shape(inputs)[0]

        original_sentences = tf.cast(inputs, dtype=tf.float32)
        splits = tf.split(original_sentences, 2, axis=1)
        disordered_sentences = tf.concat([splits[1], splits[0]], axis=1)

        p = tf.tile([[.5, .5]], [batch_size, 1])
        mask = tf.cast(1 - tf.random.categorical(tf.math.log(p), 1), dtype=tf.float32)

        cat_mask = tf.concat([mask, 1 - mask], axis=1)

        if len(tf.shape(inputs)) == 3:
            mask = mask[..., None]
        sop = original_sentences * (1 - mask) + (mask * disordered_sentences)

        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)

        sop = is_train * sop + (1 - is_train) * original_sentences
        return sop, cat_mask


class SOP_loss(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SOP_loss, self).__init__(**kwargs)

        self.sop_model = tf.keras.layers.Dense(2)

    def call(self, inputs):
        sop_mask, activity = inputs

        sop = self.sop_model(activity)
        av_sop = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(sop)
        sop_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=.1)(sop_mask, av_sop)
        self.add_loss(.01 * sop_loss)

        return activity
